#!/usr/bin/env python
# coding=utf-8
"""
@Author : yonas
@File   : main.py
"""
"""
改自T2T框架中的transformer
"""
import sys
# sys.path.append('.')

import tensorflow as tf
import learning_rate
import dataset_gen
from dataset_gen import TextEncoder
import devices
import optimizer
from util import hparams
import time
import numpy as np
import os
import platform, re
import utils
import jieba
import jieba.posseg as posseg

# import process_pre

IS_LINUX = 'Linux' in platform.system()
if IS_LINUX:
    import readline


def mask_nonzero(labels):
    """ mask: Assign weight 1.0(true)
        nonmask: Assign weight 0.0(false)
        mask if value is not 0.0
    """
    return tf.to_float(tf.not_equal(labels, 0))


def mask_nonpad_from_embedding(emb):
    """ emb: [batch,length,embed]
        Assign 1.0(true) for no pad, 0.0(false) for pad(id=0, all emb element is 0)
        return [batch,length]
    """
    return mask_nonzero(tf.reduce_sum(tf.abs(emb), axis=-1))


def length_from_embedding(emb):
    """ emb: [batch,length,embed]
        return [batch]
    """
    length = tf.reduce_sum(mask_nonpad_from_embedding(emb), axis=-1)
    length = tf.cast(length, tf.int32)
    return length


def length_from_ids(ids):
    """ ids: [batch,length,1]
        return [batch]
    """
    weight_ids = mask_nonzero(ids)
    length = tf.reduce_sum(weight_ids, axis=[1, 2])
    length = tf.cast(length, tf.int32)
    return length


def shift_right(x, pad_value=None):
    """ shift the second dim of x right by one
        decoder中对target进行右偏一位作为decode_input
        x [batch,length,embed]
        pad_value [tile_batch,1,pad_embed]
    """
    if pad_value:
        shifted_x = tf.concat([pad_value, x], axis=1)[:, :-1, :]  # length维度左边补pad_embed
    else:
        shifted_x = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]  # length维度左边补pad_embed [0,0...]
    return shifted_x


def shape_list(x):
    """ return list of dims, statically where possible """
    x = tf.convert_to_tensor(x)
    # if unknown rank, return dynamic shape 如果秩都不知道
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = shape[i] if static[i] is None else static[i]
        ret.append(dim)
    return ret


def cast_like(x, y):
    """ cast x to y's dtype, if necessary """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x
    else:
        return tf.cast(x, y.dtype)


def log_prob_from_logits(logits, axis=-1):
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=True)


def sample_with_temperature(logits, temperature):
    """ 0.0:argmax 1.0:sampling >1.0:random """
    # logits [batch,length,vocab]
    # ret [batch,length]
    if temperature == 0.0:
        return tf.argmax(logits, axis=-1)
    else:
        assert temperature > 0.0
        logits_shape = shape_list(logits)
        reshape_logits = tf.reshape(logits, [-1, logits_shape[-1]]) / temperature
        choices = tf.multinomial(reshape_logits, 1)  # 仅采样1个 该方式只支持2-D
        choices = tf.reshape(choices, logits_shape[:-1])
        return choices


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """封装dropout函数,broadcast_dims对应dropout的noise_shape"""
    assert 'noise_shape' not in kwargs
    if broadcast_dims:
        x_shape = tf.shape(x)
        ndims = len(x.get_shape())
        # allow dim like -1 as well
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
        kwargs['noise_shape'] = [1 if i in broadcast_dims else x_shape[i] for i in range(ndims)]  # 类似[1,length,hidden]
    return tf.nn.dropout(x, keep_prob, **kwargs)


def dropout_no_scaling(x, keep_prob):
    """ 不进行放缩的drop, 用以在token上 """
    if keep_prob == 1.0:
        return x
    mask = tf.less(tf.random_uniform(tf.shape(x)), keep_prob)
    mask = cast_like(mask, x)
    return x * mask


def layer_norm(x, epsilon=1e-06):
    """ layer norm """
    filters = shape_list(x)[-1]
    with tf.variable_scope('layer_norm', values=[x], reuse=None):
        scale = tf.get_variable('layer_norm_scale', [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable('layer_norm_bias', [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]  # 写法独特

    mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # mu
    variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)  # sigma
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)  # (x-mu)*sigma
    ret = norm_x * scale + bias
    return ret


def layer_prepostprocess(previous_value, x, sequence, dropout_rate, dropout_broadcast_dims=None):
    """ apply a sequence of function to the input or output of a layer
        a: add previous_value
        n: apply normalization
        d: apply dropout
        for example, if sequence=='dna', then the output is: previous_value + normalize(dropout(x))
    """
    with tf.variable_scope('layer_prepostprocess'):
        if sequence == 'none':
            return x
        for c in sequence:
            if c == 'a':
                x += previous_value  # residual
            elif c == 'n':
                x = layer_norm(x)  # LN
            else:
                assert c == 'd', 'unknown sequence step %s' % c
                x = dropout_with_broadcast_dims(x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)  # dropout
        return x


def layer_preprocess(layer_input, hparams):
    assert 'a' not in hparams.layer_preprocess_sequence, 'no residual connections allowed in preprocess sequence'
    return layer_prepostprocess(None, layer_input, sequence=hparams.layer_preprocess_sequence,
                                dropout_rate=hparams.layer_prepostprocess_dropout)


def layer_postprecess(layer_input, layer_output, hparams):
    return layer_prepostprocess(layer_input, layer_output, sequence=hparams.layer_postprocess_sequence,
                                dropout_rate=hparams.layer_prepostprocess_dropout)


def split_heads(x, num_heads):
    """ x [batch,length,hidden]
        ret [batch,num_heads,length,hidden/num_heads]
    """
    x_shape = shape_list(x)
    last_dim = x_shape[-1]
    if isinstance(last_dim, int) and isinstance(num_heads, int):
        assert last_dim % num_heads == 0
    x = tf.reshape(x, x_shape[:-1] + [num_heads, last_dim // num_heads])
    x = tf.transpose(x, [0, 2, 1, 3])
    return x


def combine_heads(x):
    """ Inverse of split_heads
        x [batch,num_heads,length,hidden/num_head]
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    x = tf.reshape(x, x_shape[:-2] + [a * b])
    return x


def compute_qkv(query_antecedent, memory_antecedent, total_key_depth, total_value_depth):
    """total_depth 包括了所有head的depth"""
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = tf.layers.dense(query_antecedent, total_key_depth, use_bias=False, name='q')
    k = tf.layers.dense(memory_antecedent, total_key_depth, use_bias=False, name='k')
    v = tf.layers.dense(memory_antecedent, total_value_depth, use_bias=False, name='v')
    return q, k, v


def dot_product_attention(q, k, v, bias, dropout_rate=0.0, dropout_broadcast_dims=None, name='dot_product_attention'):
    """ """
    with tf.variable_scope(name, values=[q, k, v]):
        logits = tf.matmul(q, k, transpose_b=True)  # [batch,num_heads,length_q,length_kv]
        if bias is not None:
            bias = cast_like(bias, logits)
        logits += bias
        weights = tf.nn.softmax(logits, name='attention_weight')  # [batch,num_heads,length_q,length_kv]
        if dropout_rate != 0:
            # drop out attention links for each head
            weights = dropout_with_broadcast_dims(weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        # v [batch,num_heads,length_kv,hidden/num_heads]
        ret = tf.matmul(weights, v)  # [batch,num_heads,length_q,hidden/num_heads]
        return ret


def multihead_attention(query_antecedent, memory_antecedent, bias, total_key_depth, total_value_depth, output_depth,
                        num_heads,
                        dropout_rate, cache=None, name='multihead_attention', dropout_broadcast_dims=None):
    """ """
    assert total_key_depth % num_heads == 0
    assert total_value_depth % num_heads == 0

    with tf.variable_scope(name, values=[query_antecedent, memory_antecedent]):
        if cache is None or memory_antecedent is None:  # training or self_attention in inferring
            q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth, total_value_depth)
        if cache is not None:  # inferring时有cache, 此时query_antecedent均为[batch,1,hidden]
            assert bias is not None, 'Bias required for caching'

            if memory_antecedent is not None:  # encode-decode attention 使用cache
                q = tf.layers.dense(query_antecedent, total_key_depth, use_bias=False, name='q')
                k = cache['k_encdec']
                v = cache['v_encdec']
            else:  # decode self_attention 得到k,v需存到cache
                k = split_heads(k, num_heads)  # [batch,num_heads,length,hidden/num_heads]
                v = split_heads(v, num_heads)
                k = cache['k'] = tf.concat([cache['k'], k], axis=2)
                v = cache['v'] = tf.concat([cache['v'], v], axis=2)
        q = split_heads(q, num_heads)
        if cache is None:
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q = q * key_depth_per_head ** -0.5  # scale

        x = dot_product_attention(q, k, v, bias, dropout_rate, dropout_broadcast_dims=dropout_broadcast_dims)

        x = combine_heads(x)

        x.set_shape(x.shape.as_list()[:-1] + [total_key_depth])  # set last dim specifically

        x = tf.layers.dense(x, output_depth, use_bias=False, name='output_transform')
        return x


def attention_bias_lower_triangle(length):
    """ 下三角矩阵 """
    band = tf.matrix_band_part(tf.ones([length, length]), -1, 0)  # [length,length] 下三角矩阵,下三角均为1,上三角均为0
    # [[1,0,0],
    #  [1,1,0],
    #  [1,1,1]] float
    band = tf.reshape(band, [1, 1, length, length])
    band = -1e9 * (1.0 - band)
    # [[0,-1e9,-1e9],
    #  [0,0,-1e9],
    #  [0,0,0]]
    return band


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """ use for position embedding """
    length = shape_list(x)[1]
    hidden_size = shape_list(x)[2]
    signal = get_timing_signal_1d(length, hidden_size, min_timescale, max_timescale, start_index)
    return x + signal


def get_timing_signal_1d(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """ use for calculate position embedding """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = hidden_size // 2
    log_timescales_increment = np.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescales_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal


def transformer_ffn_layer(x, hparams, pad_mask=None):
    original_shape = shape_list(x)  # [batch,length,hidden]

    if pad_mask is not None:
        """ remove pad """
        flat_pad_mask = tf.reshape(pad_mask, [-1])  # [batch*length]
        flat_nonpad_ids = tf.to_int32(tf.where(tf.equal(flat_pad_mask, 0)))

        # flat x
        x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))  # [batch*length,hidden]
        # remove pad
        x = tf.gather_nd(x, flat_nonpad_ids)  # [batch*length-,hidden]

    h = tf.layers.dense(x, hparams.filter_size, use_bias=True, activation=tf.nn.relu, name='conv1')

    if hparams.relu_dropout != 0.:
        h = dropout_with_broadcast_dims(h, 1.0 - hparams.relu_dropout, broadcast_dims=None)

    o = tf.layers.dense(h, hparams.hidden_size, activation=None, use_bias=True, name='conv2')

    if pad_mask is not None:
        """ restore pad """
        o = tf.scatter_nd(  # 将updates中对应的值填充到indices指定的索引中，空的位置会用0代替，刚好代表pad
            indices=flat_nonpad_ids,
            updates=o,
            shape=tf.concat([tf.shape(flat_pad_mask)[:1], tf.shape(o)[1:]], axis=0)
        )

        o = tf.reshape(o, original_shape)
    return o


def transformer_encoder(encoder_input, encoder_pad_mask, encoder_self_attention_bias):
    x = encoder_input
    with tf.variable_scope('encoder'):
        for layer in range(hparams.num_encoder_layers):
            with tf.variable_scope('layer_%d' % layer):
                with tf.variable_scope('self_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, hparams),
                        None,  # self attention
                        encoder_self_attention_bias,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout)
                    x = layer_postprecess(x, y, hparams)
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        layer_preprocess(x, hparams),
                        hparams,
                        pad_mask=encoder_pad_mask)
                    x = layer_postprecess(x, y, hparams)
        encoder_output = layer_preprocess(x, hparams)
    return encoder_output


def transformer_decoder(decoder_input, encoder_output, decoder_pad_mask, decoder_self_attention_bias, encoder_decoder_attention_bias,
                        cache=None):
    x = decoder_input
    with tf.variable_scope('decoder'):
        for layer in range(hparams.num_decoder_layers):
            layer_name = 'layer_%d' % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope('self_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, hparams),
                        None,
                        decoder_self_attention_bias,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        cache=layer_cache)
                    x = layer_postprecess(x, y, hparams)
                with tf.variable_scope('encdec_attention'):
                    y = multihead_attention(
                        layer_preprocess(x, hparams),
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        cache=layer_cache)
                    x = layer_postprecess(x, y, hparams)
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        layer_preprocess(x, hparams),
                        hparams,
                        pad_mask=decoder_pad_mask)
                    x = layer_postprecess(x, y, hparams)
        decoder_output = layer_preprocess(x, hparams)
    return decoder_output


class Transformer(object):
    def __init__(self, train_data_dir=None, dp=None, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            assert train_data_dir
            self.train_data_dir = train_data_dir
            """ train """
            if dp is None:
                num_gpus = 1
                if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2:
                    num_gpus = 2
                dp = devices.data_parallelism(
                    sync=False,
                    worker_id=0,
                    worker_gpu=num_gpus,
                    worker_replicas=1,
                    worker_job='/job:localhost',
                    ps_gpu=0,
                    ps_replicas=0,
                    ps_job='/job:ps',
                    daisy_chain_variables=True)
            self.dp = dp

            # dataset pipeline
            dataset = dataset_gen.get_dataset(self.train_data_dir, hparams.batch_size, dp.n)
            iterator = dataset.make_one_shot_iterator()
            example = iterator.get_next()
            msg = example['msg']  # [batch,length]
            label = example['label']  # [batch,1]
            pos = example['pos']  # [batch,length]
            seg = example['seg']  # [batch,length]
            ner = example['ner']  # [batch,length]

            # expand to 3D
            msg = tf.expand_dims(msg, axis=-1)
            pos = tf.expand_dims(pos, axis=-1)
            seg = tf.expand_dims(seg, axis=-1)
            ner = tf.expand_dims(ner, axis=-1)

            # 初始化
            tf.get_variable_scope().set_initializer(optimizer.get_variable_initializer(hparams))
            lr = learning_rate.learning_rate_schedule(hparams)

            """ 一、支持分布式数据并行 """
            ## shard inputs and targets
            # inputs_list = self.dp(tf.identity, tf.split(msg, self.dp.n, 0))
            # targets_list = self.dp(tf.identity, tf.split(label, self.dp.n, 0))
            #
            # loss_list = self.dp(self.train, inputs_list, targets_list)
            # self.loss = tf.add_n(loss_list) / self.dp.n

            """ 二、非分布式 数据不并行 """
            self.loss_dict, self.metrics = self.train(msg, label, pos, seg, ner)
            self.loss = self.loss_dict['total_loss']

            self.train_op = optimizer.optimize(self.loss, lr, hparams, use_tpu=False)

        else:
            """ infer """
            self.msg = tf.placeholder(tf.int32, shape=[None, None], name='inputs')  # [batch,length]
            self.pos = tf.placeholder(tf.int32, shape=[None, None], name='inputs')  # [batch,length]
            self.seg = tf.placeholder(tf.int32, shape=[None, None], name='inputs')  # [batch,length]
            msg = tf.expand_dims(self.msg, axis=-1)  # [batch,length,1]
            pos = tf.expand_dims(self.pos, axis=-1)  # [batch,length,1]
            seg = tf.expand_dims(self.seg, axis=-1)  # [batch,length,1]
            self.result = self.infer(msg, pos, seg)

    def train(self, msg, label, pos, seg, ner):
        """ """
        """ embedding """
        # msg/pos/seg/ner [batch,length,1] label [batch,1]
        ner = tf.cast(ner, dtype=tf.float32)

        embedding_msg = bottom(msg, 'embedding', reuse=tf.AUTO_REUSE)  # [batch,length,embed]
        embedding_pos = embedding(pos, 'pos_embedding', 26, reuse=tf.AUTO_REUSE)  # [batch,length,embed]
        embedding_seg = embedding(seg, 'seg_embedding', 4, reuse=tf.AUTO_REUSE)  # [batch,length,embed]

        """ encoder """
        encoder_input = embedding_msg

        mask_nonpad = mask_nonpad_from_embedding(embedding_msg)  # [batch,length] 1 for nonpad; 0 for pad

        encoder_pad_mask = 1. - mask_nonpad  # [batch,length] 1 for pad; 0 for nonpad
        encoder_self_attention_bias = encoder_pad_mask * -1e9  # attention mask
        encoder_self_attention_bias = tf.expand_dims(tf.expand_dims(encoder_self_attention_bias, axis=1), axis=1)  # [batch,1,1,length]

        encoder_input = add_timing_signal_1d(encoder_input)  # add position embedding

        encoder_input += embedding_pos
        encoder_input += embedding_seg
        # encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)  # drop

        encoder_output = transformer_encoder(encoder_input, encoder_pad_mask, encoder_self_attention_bias)

        """ loss """
        # ner loss #
        ner_layer_logits = tf.layers.dense(encoder_output, 1, activation=None, use_bias=True)  # [batch,length,1]
        # ner [batch,length,1]
        ner_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=ner_layer_logits, labels=ner)  # [batch,length,1]
        ner_loss = tf.squeeze(ner_loss, axis=-1)  # [batch,length]

        ner_loss = ner_loss * mask_nonpad  # [batch,length]
        ner_loss = tf.reduce_mean(ner_loss)  # [scalar]

        ner_layer_probs = tf.nn.sigmoid(ner_layer_logits)  # [batch,length,1]

        ner_layer_preds = tf.where(ner_layer_probs < 0.5, tf.zeros_like(ner_layer_probs, dtype=tf.float32), tf.ones_like(ner_layer_probs, dtype=tf.float32))

        length_nonpad = tf.reduce_sum(tf.squeeze(mask_nonpad, axis=-1), axis=1)  # [batch]
        ner_acc_nu = tf.to_float(tf.equal(ner_layer_preds, ner)) * mask_nonpad  # [batch,length,1]
        ner_acc_nu = tf.reduce_sum(tf.squeeze(ner_acc_nu, axis=-1), axis=-1)  # [batch]
        ner_acc_deno = length_nonpad  # [batch]
        ner_acc = tf.reduce_mean(ner_acc_nu / ner_acc_deno, axis=0)  # [scalar]

        # cls loss #
        mask_nonpad_ = tf.expand_dims(mask_nonpad, axis=-1)  # [batch,length,1]
        cls_output = ner * mask_nonpad_ * encoder_output  # [batch,length,hidden]
        cls_output = tf.reduce_mean(cls_output, axis=1)  # [batch,hidden]

        cls_logits = tf.layers.dense(cls_output, 660, activation=None, use_bias=True)  # [batch,vocab]

        label = tf.squeeze(label)  # [batch]
        onehot_label = tf.one_hot(label, depth=660)  # [batch,vocab]

        cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=cls_logits, labels=onehot_label)  # [batch]
        cls_loss = tf.reduce_mean(cls_loss)  # [scalar]

        total_loss = ner_loss + cls_loss

        cls_pred = tf.argmax(cls_logits, axis=-1)
        cls_acc = tf.reduce_mean(tf.to_float(tf.equal(label, cls_pred)))

        metrics = {'cls_acc': cls_acc,
                   'ner_acc': ner_acc}

        loss_dict = {'ner_loss': ner_loss,
                     'cls_loss': cls_loss,
                     'total_loss': total_loss,
                     }

        return loss_dict, metrics

        # _, pre_topk_label = tf.nn.top_k(logits, k=5)  # [batch,topk]
        # pre_topk_label = tf.cast(pre_topk_label, dtype=tf.int32)
        #
        # batch_size = tf.shape(logits)[0]
        #
        # flags = tf.equal(pre_topk_label, tf.expand_dims(label, axis=-1))  # [batch,topk]
        # acc_topk_list = tf.zeros([batch_size, 0], dtype=tf.int32)  # [batch, topk]
        # acc_topk_list = tf.concat([acc_topk_list, tf.to_int32(tf.reduce_any(flags[:, :1], axis=-1, keep_dims=True))], axis=-1)
        # acc_topk_list = tf.concat([acc_topk_list, tf.to_int32(tf.reduce_any(flags[:, :2], axis=-1, keep_dims=True))], axis=-1)
        # acc_topk_list = tf.concat([acc_topk_list, tf.to_int32(tf.reduce_any(flags[:, :3], axis=-1, keep_dims=True))], axis=-1)
        # acc_topk_list = tf.concat([acc_topk_list, tf.to_int32(tf.reduce_any(flags[:, :4], axis=-1, keep_dims=True))], axis=-1)
        # acc_topk_list = tf.concat([acc_topk_list, tf.to_int32(tf.reduce_any(flags[:, :5], axis=-1, keep_dims=True))], axis=-1)
        #
        # self.acc_topk_list = tf.reduce_sum(acc_topk_list, axis=0) / batch_size  # [topk]

        # acc_topk = tf.to_int32(tf.reduce_any(tf.equal(pre_topk_label, tf.expand_dims(label, axis=-1)), axis=-1))  # [batch]
        # self.acc_topk = tf.reduce_sum(acc_topk) / tf.shape(acc_topk)[0]
        #
        # # top1 acc
        # acc = tf.to_int32(tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), label))  # [batch]
        # self.acc = tf.reduce_sum(acc) / tf.shape(acc)[0]


    def infer(self, msg, pos, seg):
        """ embedding """
        # msg/pos/seg [batch,length,1]
        embedding_msg = bottom(msg, 'embedding', reuse=tf.AUTO_REUSE)  # [batch,length,embed]
        embedding_pos = embedding(pos, 'pos_embedding', 26, reuse=tf.AUTO_REUSE)  # [batch,length,embed]
        embedding_seg = embedding(seg, 'seg_embedding', 4, reuse=tf.AUTO_REUSE)  # [batch,length,embed]

        """ encoder """
        encoder_input = embedding_msg

        mask_nonpad = mask_nonpad_from_embedding(embedding_msg)  # [batch,length] 1 for nonpad; 0 for pad

        encoder_pad_mask = 1. - mask_nonpad_from_embedding(embedding_msg)  # [batch,length] 1 for pad; 0 for nonpad
        encoder_self_attention_bias = encoder_pad_mask * -1e9  # attention mask
        encoder_self_attention_bias = tf.expand_dims(tf.expand_dims(encoder_self_attention_bias, axis=1), axis=1)  # [batch,1,1,length]

        encoder_input = add_timing_signal_1d(encoder_input)  # add position embedding

        encoder_input += embedding_pos
        encoder_input += embedding_seg
        # encoder_input = tf.nn.dropout(encoder_input, 1.0-hparams.layer_prepostprocess_dropout)  # drop

        encoder_output = transformer_encoder(encoder_input, encoder_pad_mask, encoder_self_attention_bias)

        # ner loss #
        ner_layer_logits = tf.layers.dense(encoder_output, 1, activation=None, use_bias=True)  # [batch,length,1]
        # ner [batch,length,1]
        ner_layer_alpha = tf.nn.sigmoid(ner_layer_logits)
        # ner_layer_alpha = tf.squeeze(ner_layer_alpha, axis=-1)  # [batch,length]
        one = tf.ones_like(ner_layer_alpha, dtype=tf.float32)
        zero = tf.zeros_like(ner_layer_alpha, dtype=tf.float32)
        ner = tf.where(ner_layer_alpha < 0.5, zero, one)

        # cls loss #
        mask_nonpad_ = tf.expand_dims(mask_nonpad, axis=-1)  # [batch,length,1]
        cls_output = ner * mask_nonpad_ * encoder_output  # [batch,length,hidden]
        cls_output = tf.reduce_mean(cls_output, axis=1)  # [batch,hidden]

        cls_logits = tf.layers.dense(cls_output, 660, activation=None, use_bias=True)  # [batch,vocab]

        cls_probs = tf.nn.softmax(cls_logits)

        """ top and loss """
        # pred_label_id = tf.argmax(logits, axis=-1)  # [batch]
        # pred_prob = tf.reduce_max(cls_probs, axis=-1)

        # topk
        pre_topk_probs, pre_topk_label = tf.nn.top_k(cls_probs, k=5)
        return {'pred_topk_labels': pre_topk_label, 'pred_topk_probs': pre_topk_probs, 'probs': cls_probs, 'ner': ner, 'ner_layer_alpha': ner_layer_alpha}


def bottom(ids, name, reuse):
    """ embedding """
    # ids [batch,length,1]
    with tf.variable_scope(name, reuse=reuse):
        var = tf.get_variable('weights', [hparams.vocab_size, hparams.hidden_size],  # [vocab,hidden]
                              initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5))

    # lookup
    ids = dropout_no_scaling(ids, 1.0 - hparams.symbol_dropout)  # 随机将部分id变为0,相当于将单词变为pad
    embedding = tf.gather(var, ids)  # [batch,length,1,hidden]
    embedding = tf.squeeze(embedding, axis=2)  # [batch,length,hidden]
    if hparams.multiply_embedding_mode == 'sqart_depth':
        embedding *= hparams.hidden_size ** 0.5
    embedding = embedding * tf.to_float(tf.not_equal(ids, 0))  # 将pad(id=0)的emb变为[0,0,...]
    return embedding


def embedding(ids, name, vocab_size, reuse):
    """ embedding """
    # ids [batch, length, 1]
    with tf.variable_scope(name, reuse=reuse):
        var = tf.get_variable('weights', [vocab_size, hparams.hidden_size],  # [vocab,hidden]
                              initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5))
    # lookup
    # ids = dropout_no_scaling(ids, 1.0 - hparams.symbol_dropout)  # 随机将部分id变为0,相当于将单词变为pad
    embedding = tf.gather(var, ids)  # [batch,length,1,hidden]
    embedding = tf.squeeze(embedding, axis=2)  # [batch,length,hidden]
    if hparams.multiply_embedding_mode == 'sqart_depth':
        embedding *= hparams.hidden_size ** 0.5
    embedding = embedding * tf.to_float(tf.not_equal(ids, 0))  # 将pad(id=0)的emb变为[0,0,...]
    return embedding


def train():
    model = Transformer(train_data_dir=hparams.train_data_dir)
    tf.logging.info('Graph loaded')

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    global_step = tf.train.get_or_create_global_step()

    with tf.train.MonitoredTrainingSession(
            save_checkpoint_secs=None,
            # save_checkpoint_steps=None,
            checkpoint_dir=hparams.model_dir,
            log_step_count_steps=None,  # 隔几步打印 global_step/sec
            config=session_config,
            hooks=[tf.train.StopAtStepHook(last_step=hparams.max_steps),
                   # tf.train.ProfilerHook(save_step=100),
                   tf.train.LoggingTensorHook(tensors={'loss': model.loss,
                                                       'ner_loss': model.loss_dict['ner_loss'],
                                                       'cls_loss': model.loss_dict['cls_loss'],
                                                       'cls_acc': model.metrics['cls_acc'],
                                                       'ner_acc': model.metrics['ner_acc'],
                                                       'step': global_step}, every_n_iter=100),
                   tf.train.CheckpointSaverHook(hparams.model_dir, save_steps=hparams.save_steps,
                                                saver=tf.train.Saver(max_to_keep=hparams.max_to_keep, save_relative_paths=True))
                   ]
    ) as sess:
        while not sess.should_stop():
            _, metrics, g_step= sess.run([model.train_op, model.metrics, global_step])
            # 打印loss功能交由LoggingTensorHook



def evaluate():
    model = Transformer(train_data_dir=hparams.eval_data_dir)
    tf.logging.info('Graph loaded')

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
        gpu_options=tf.GPUOptions(allow_growth=True),
    )
    global_step = tf.train.get_or_create_global_step()

    with tf.train.MonitoredTrainingSession(
            save_checkpoint_secs=None,
            # save_checkpoint_steps=None,
            checkpoint_dir=hparams.model_dir,
            log_step_count_steps=None,  # 隔几步打印 global_step/sec
            config=session_config,
            hooks=[tf.train.StopAtStepHook(last_step=hparams.max_steps),
                   # tf.train.ProfilerHook(save_step=100),
                   tf.train.LoggingTensorHook(tensors={'loss': model.loss,
                                                       # 'pre_acc': model.acc,
                                                       # 'pre_acc_topk': model.acc_topk,
                                                       'pre_acc_topk_list': model.acc_topk_list,
                                                       'step': global_step}, every_n_iter=100),
                   ]
    ) as sess:
        while not sess.should_stop():
            _ = sess.run([model.loss])
            # 打印loss功能交由LoggingTensorHook


def get_pos_and_seg(text):
    word_list = []
    pos_list = []
    seg_list = []
    seg_res = []
    for token in posseg.cut(text, HMM=False):
        seg_res.append(token.word)
        words = list(token.word)
        length = len(words)
        pos = [token.flag[:1]] * length  # 截取词性标记首字母
        if len(words) == 1:
            seg = ['S']
        else:
            seg = ['B'] + ['I'] * (length - 2) + ['E']
        word_list.extend(words)
        pos_list.extend(pos)
        seg_list.extend(seg)
    seg_res = ' '.join(seg_res)
    return word_list, pos_list, seg_list, seg_res


def infer():
    L4id2name = utils.file2dict('L4.txt')

    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    text_encoder = TextEncoder('worddict.txt', replace_oov='<UNK>')
    label_encoder = TextEncoder('labeldict.txt')
    pos_encoder = TextEncoder('posdict.txt')
    ner_encoder = TextEncoder('nerdict.txt')
    seg_encoder = TextEncoder('segdict.txt')

    model = Transformer(mode='infer')
    tf.logging.info('Graph Loaded')
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, tf.train.latest_checkpoint(hparams.model_dir))
    tf.logging.info('Restored!')

    time_record = []

    def print_result(msg):
        msg_ids = text_encoder.encode(msg, as_list=True)
        word_list, pos_list, seg_list, seg_res = get_pos_and_seg(msg)
        pos_ids = pos_encoder.encode(pos_list, as_list=True)
        seg_ids = seg_encoder.encode(seg_list, as_list=True)
        assert len(msg_ids) == len(pos_ids) == len(seg_ids)

        msg_ids = np.array([msg_ids], dtype=np.int32)  # [1,length]
        pos_ids = np.array([pos_ids], dtype=np.int32)  # [1,length]
        seg_ids = np.array([seg_ids], dtype=np.int32)  # [1,length]

        start_time = time.time()
        result = sess.run(model.result, {model.msg: msg_ids,
                                         model.pos: pos_ids,
                                         model.seg: seg_ids,
                                         })
        elapsed = time.time() - start_time
        time_record.append(elapsed)
        print('elapsed:{:.0f}ms'.format(elapsed * 1000))
        if len(time_record) > 3:
            print('avg_elapsed:{:.0f}ms'.format(sum(time_record[3:]) / len(time_record[3:]) * 1000))
        # {'pred_topk_labels': pre_topk_label, 'pred_topk_probs': pre_topk_probs, 'probs': cls_probs, 'ner': ner, 'ner_layer_alpha': ner_layer_alpha}
        print('seg_res', '\t', seg_res)
        print('seg    ', '\t', pos_list)
        print('pos    ', '\t', seg_list)

        # batch_size = 1
        pred_topk_labels = result['pred_topk_labels'][0]
        pred_topk_probs = result['pred_topk_probs'][0]
        probs = result['probs'][0]
        ner = result['ner'][0]
        ner_layer_alpha = result['ner_layer_alpha'][0]

        ner = np.squeeze(ner, axis=-1).tolist()
        ner_str = ' '.join([f'{ele:.0f}' for ele in ner])
        ner_layer_alpha = np.squeeze(ner_layer_alpha, axis=-1).tolist()
        ner_prob = ' '.join([f'{ele:.2f}' for ele in ner_layer_alpha])
        print('ner    ', '\t', ner_str)
        print('ner_prob', '\t', ner_prob)

        key_words = [msg[i] for i in range(len(msg)) if ner[i] == 1]
        print('key_word', '\t', ' '.join(key_words))
        for i in range(5):
            label = label_encoder.decode([pred_topk_labels[i]])
            label = L4id2name[label]
            prob = pred_topk_probs[i]
            print(label, '\t', prob)

    print_result('微信群发的红包发不了什么原因？')
    while True:
        msg = input('请输入')
        msg = norm(msg)
        # print(msg)
        print_result(msg)


def infer_from_file(in_f='testset.txt', out_f='testset_res.txt'):
    L4id2name = utils.file2dict('L4.txt')

    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    text_encoder = TextEncoder('worddict.txt', replace_oov='<UNK>')
    label_encoder = TextEncoder('labeldict.txt')
    pos_encoder = TextEncoder('posdict.txt')
    ner_encoder = TextEncoder('nerdict.txt')
    seg_encoder = TextEncoder('segdict.txt')

    model = Transformer(mode='infer')
    tf.logging.info('Graph Loaded')
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, tf.train.latest_checkpoint(hparams.model_dir))
    tf.logging.info('Restored!')

    msg_list = utils.file2list(in_f)
    result_list = []

    for msg in msg_list:
        if not msg:
            result_list.append('')
            continue

        ori_msg = msg
        msg = norm(msg)
        msg_ids = text_encoder.encode(msg, as_list=True)
        word_list, pos_list, seg_list, seg_res = get_pos_and_seg(msg)
        pos_ids = pos_encoder.encode(pos_list, as_list=True)
        seg_ids = seg_encoder.encode(seg_list, as_list=True)
        assert len(msg_ids) == len(pos_ids) == len(seg_ids)

        msg_ids = np.array([msg_ids], dtype=np.int32)  # [1,length]
        pos_ids = np.array([pos_ids], dtype=np.int32)  # [1,length]
        seg_ids = np.array([seg_ids], dtype=np.int32)  # [1,length]

        result = sess.run(model.result, {model.msg: msg_ids,
                                         model.pos: pos_ids,
                                         model.seg: seg_ids,
                                         })
        # {'pred_topk_labels': pre_topk_label, 'pred_topk_probs': pre_topk_probs, 'probs': cls_probs, 'ner': ner, 'ner_layer_alpha': ner_layer_alpha}
        pred_topk_labels = result['pred_topk_labels'][0]
        pred_topk_probs = result['pred_topk_probs'][0]
        probs = result['probs'][0]
        ner = result['ner'][0]
        ner_layer_alpha = result['ner_layer_alpha'][0]

        ner = np.squeeze(ner, axis=-1).tolist()  # 1/0
        ner_str = ' '.join([f'{ele:.0f}' for ele in ner])
        ner_layer_alpha = np.squeeze(ner_layer_alpha, axis=-1).tolist()
        ner_prob = ' '.join([f'{ele:.2f}' for ele in ner_layer_alpha])

        key_words = [msg[i] for i in range(len(msg)) if ner[i] == 1]
        pred_label = L4id2name[label_encoder.decode([pred_topk_labels[0]])]
        prob = pred_topk_probs[0]

        out = f'{ori_msg}\t{pred_label}({prob:.2f})\t{" ".join(key_words)}'
        result_list.append(out)

    utils.list2file(out_f, result_list)


def batch_wrap(gen, batch_size=64):
    batch_bucket = []
    for ele in gen:
        batch_bucket.append(ele)
        if len(batch_bucket) == batch_size:
            yield batch_bucket
            batch_bucket = []
    if batch_bucket:
        yield batch_bucket


def infer_evaluate(test_file, max_count=1000):
    """
    验证测试文件的topk准确率
    """
    L4id2name = utils.file2dict('L4.txt')

    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    text_encoder = TextEncoder('worddict.txt', replace_oov='<UNK>')
    label_encoder = TextEncoder('labeldict.txt')
    pos_encoder = TextEncoder('posdict.txt')
    ner_encoder = TextEncoder('nerdict.txt')
    seg_encoder = TextEncoder('segdict.txt')

    model = Transformer(mode='infer')
    tf.logging.info('Graph Loaded')
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, tf.train.latest_checkpoint(hparams.model_dir))
    tf.logging.info('Restored!')

    eval_data = []
    f = open(test_file, 'r', encoding='U8')
    while True:
        line = f.readline().strip()
        if not line:
            break
        try:
            items = line.split(':')
            msg = norm(items[-3])
            label = items[-4]
            ner = items[-2]
        except:
            print(line)
            continue

        if label in ['4000007', '4000012', '4000016']:
            continue
        msg_ids = text_encoder.encode(msg, as_list=True)
        label_id = label_encoder.encode(label, as_list=False)  # [400001]
        if label_id == []:
            4000007
            print(label)
            print(line)

        ner_ids = ner_encoder.encode(ner, as_list=False)

        word_list, pos_list, seg_list, _ = get_pos_and_seg(msg)
        pos_ids = pos_encoder.encode(pos_list, as_list=True)
        seg_ids = seg_encoder.encode(seg_list, as_list=True)
        assert len(msg_ids) == len(ner_ids) == len(pos_ids) == len(seg_ids)
        eval_data.append([msg_ids, label_id, ner_ids, pos_ids, seg_ids, msg])
    f.close()

    acc_list = []
    ner_acc_list = []

    out_file = open('eval_res.txt', 'w', encoding='U8')

    for i, (msg_ids, true_label_id, true_ner_ids, pos_ids, seg_ids, msg) in enumerate(eval_data):
        if i > max_count:
            break
        if not i % 1000:
            print('progress', i)
        try:
            msg_str = msg
            # msg_str = text_encoder.decode(msg_ids, strip_extraneous=True, keep_space=False)
            # msg_str = msg_str.replace('<UNK>', 'u')
            true_label = L4id2name[label_encoder.decode(true_label_id)]
            true_keyword = [msg_str[i] for i in range(len(msg_str)) if true_ner_ids[i] == 1]
            true_keyword = ' '.join(true_keyword)
        except:
            print(msg_str)
            print(msg)
            print(true_label_id)
            print(true_label)
            print(true_ner_ids)
            raise

        msg_ids = np.array([msg_ids], dtype=np.int32)  # [1,length]
        pos_ids = np.array([pos_ids], dtype=np.int32)  # [1,length]
        seg_ids = np.array([seg_ids], dtype=np.int32)  # [1,length]
        result = sess.run(model.result, {model.msg: msg_ids,
                                         model.pos: pos_ids,
                                         model.seg: seg_ids,
                                         })

        # batch_size = 1
        pred_topk_labels = result['pred_topk_labels'][0]
        pred_topk_probs = result['pred_topk_probs'][0]
        probs = result['probs'][0]
        ner = result['ner'][0]
        ner_layer_alpha = result['ner_layer_alpha'][0]

        ner = np.squeeze(ner, axis=-1)
        ner_layer_alpha = np.squeeze(ner_layer_alpha, axis=-1)

        pred_label = pred_topk_labels[0]

        pred_label_str = L4id2name[label_encoder.decode([pred_label])]
        pred_keyword = [msg_str[i] for i in range(len(msg_str)) if ner[i] == 1]
        pred_keyword = ' '.join(pred_keyword)

        out_file.write(msg_str + '\t' + true_label + '\t' + true_keyword + '\n')
        out_file.write(msg_str + '\t' + pred_label_str + '\t' + pred_keyword + '\n')
        out_file.write('\n')

        if pred_label == true_label_id:
            acc_list.append(1)
        else:
            acc_list.append(0)

        true_ner_ids = np.array(true_ner_ids).astype(np.int32)
        ner = ner.astype(np.int32)
        ner_acc_list.append(np.mean(true_ner_ids == ner))

    cls_acc = np.mean(acc_list)
    ner_acc_list = np.mean(ner_acc_list)

    print('cls_acc', cls_acc)
    print('ner_acc_list', ner_acc_list)
    out_file.close()


def norm(msg):
    msg = msg.replace('，', ',')
    # msg = msg.replace('。', '.')
    msg = msg.replace('！', '!')
    msg = msg.replace('？', '?')
    msg = msg.replace('～', '~')
    msg = msg.replace('；', ',')
    msg = msg.replace(';', ',')
    msg = msg.replace('：', ',')
    msg = msg.replace(':', ',')
    msg = msg.replace('【', '(')
    msg = msg.replace('】', ')')
    msg = msg.replace('（', '(')
    msg = msg.replace('）', ')')
    msg = msg.replace('‘', ',')
    msg = msg.replace('’', ',')
    return msg


def infer_evaluate_batch(test_file, batch_size=128, max_count=1000, out_file=None, model_id=None):
    """
    验证测试文件的topk准确率（batch）
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.reset_default_graph()
    encoder = TextEncoder(hparams.worddict, replace_oov='<UNK>')
    label_encoder = TextEncoder(hparams.labeldict)

    model = Transformer(mode='infer')
    tf.logging.info('Graph Loaded')
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if not model_id:
        model_path = tf.train.latest_checkpoint(hparams.model_dir)
    else:
        model_path = hparams.model_dir + '/model.ckpt-%s' % model_id
    saver.restore(sess, model_path)
    tf.logging.info('Restored!')
    tf.logging.info('eval_file:' + test_file)
    with open(test_file, 'r', encoding='U8') as f:
        lines = [line.strip().rsplit('\t', maxsplit=2) for line in f]  # 有sessionid
    with open(hparams.labeldict, 'r', encoding='U8') as f:
        label_set = {line.strip() for line in f}

    def batch_generator(lines):
        msg_bucket, label_bucket, sessid_bucket = [], [], []
        for msg, label, sessid in lines:
            # print(msg, label)
            # 判断是否是符合的数据
            # label = re.search('\d\.【(.+?)】', label)
            # if label and len(label.groups()) > 0:
            #     label = label.groups()[0]
            # else:
            #     continue
            # if label == '超级影视vip':
            #     label = '腾讯视频'
            if label not in label_set:
                print('跳过', label)
                continue
            msg = msg.replace('\t', '')
            msg = norm(msg)
            if not msg:
                continue
            msg_bucket.append(msg)
            label_bucket.append(label)
            sessid_bucket.append(sessid)
            if len(msg_bucket) == batch_size:
                yield msg_bucket, label_bucket, sessid_bucket
                msg_bucket, label_bucket, sessid_bucket = [], [], []

    if out_file:
        fo = open(out_file, 'w', encoding='U8')
    for i, (batch_msg, batch_label, batch_sessid) in enumerate(batch_generator(lines)):
        if max_count and i > (max_count // batch_size + 1):
            break
        if not i % 5:
            print(batch_size * i)

        batch_label_id = [label_encoder.encode(label, as_list=False)[0] for label in batch_label]  # [batch]
        batch_msg_ids = [[encoder.cls_id, *encoder.encode(msg)] for msg in batch_msg]  # [batch,length]
        # print(batch_msg, '->', batch_label)
        # print(batch_msg_ids, '->', batch_label_id)
        max_length = max([len(msg_ids) for msg_ids in batch_msg_ids])
        batch_msg_ids = [msg_ids + [encoder.pad_id] * (max_length - len(msg_ids)) for msg_ids in batch_msg_ids]
        x = np.array(batch_msg_ids, dtype=np.int32)  # [batch,length]
        result = sess.run(model.result, {model.inputs: x})
        for i, pre_label_ids in enumerate(result['outputs']):  # 遍历batch
            pre_label_ids = list(pre_label_ids)
            pre_probs = list(result['scores'][i])
            pre_labels = label_encoder.decode(pre_label_ids).split(' ')
            # print(list(zip(pre_labels, pre_probs)), end='\n\n')
            topk_flags = [batch_label_id[i] in pre_label_ids[:ii + 1] for ii in range(len(pre_label_ids))]
            topk_flags = list(map(int, topk_flags))
            if out_file:
                print('\t'.join(['{}'] * 13).format(
                    batch_msg[i],
                    batch_label[i],
                    batch_sessid[i],
                    *[f'{pre_labels[j]}({pre_probs[j]:.2f})' for j in range(5)],
                    *[topk_flags[j] for j in range(5)],
                ), file=fo)
    if out_file:
        fo.close()


def stat_class(in_file):
    stat_L1_dict = {}
    with open(in_file, 'r', encoding='U8') as f:
        lines = [line.strip() for line in f]
    for line in lines:
        items = line.split('\t')
        if items[1] not in stat_L1_dict:
            stat_L1_dict[items[1]] = [items[-5:]]
        else:
            stat_L1_dict[items[1]].append(items[-5:])
    ret = []
    for key in stat_L1_dict:
        try:
            nums = len(stat_L1_dict[key])
            stat_L1_dict[key] = np.mean(np.array(stat_L1_dict[key], dtype=np.int32), axis=0).tolist()
        except:
            print(stat_L1_dict[key])
            raise
        ret.append('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}({})'.format(*stat_L1_dict[key], key, nums))
        # print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}({})'.format(*stat_L1_dict[key], key, nums))
    ret.sort(key=lambda item: item.split('\t')[0], reverse=True)
    print(*ret, sep='\n')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    tf.logging.set_verbosity(tf.logging.INFO)

    jieba.load_userdict('user_dict.txt')

    if IS_LINUX:
        hparams.worddict = 'worddict.txt'
        hparams.labeldict = 'labeldict0620.txt'
        hparams.train_data_dir = 'tfrecord/ambiscene-train*'
        hparams.train_data_dir = 'tfrecord/keywordscene-train*'
        hparams.train_data_dir = 'tfrecord/allscene-train*'
        hparams.train_data_dir = 'tfrecord/allscene_chitchat-train*'
        hparams.eval_data_dir = 'tfrecord/keywordscene-train*'
        hparams.model_dir = 'model/base_all'
        hparams.batch_size = 128  # 3479052 / 128 = 27180
        hparams.save_steps = 27180
    else:
        hparams.worddict = '../corpus/train_data/worddict.txt'
        hparams.labeldict = '../corpus/train_data/labeldict0620.txt'
        hparams.train_data_dir = 'tfrecord/keywordscene-train*'
        hparams.train_data_dir = 'tfrecord/allscene_chitchat-train*'
        hparams.eval_data_dir = 'tfrecord/keywordscene-train*'
        hparams.model_dir = 'model/test'
        hparams.batch_size = 64
        hparams.save_steps = 1000

    # 58430
    hparams.batch_size = 128  # 58430 / 128 = 457
    hparams.save_steps = 1825  # 4 epoch

    hparams.vocab_size = 5387
    hparams.max_to_keep = 100
    hparams.labeldict = 'labeldict.txt'
    hparams.train_data_dir = 'tfrecord/kw_ner*'
    # hparams.model_dir = 'model/0620/soft'
    hparams.model_dir = 'model/0717_1/'
    # hparams.activator = 'softmax'
    # hparams.activator = 'sigmoid'

    hparams.max_steps = 22850  # 50 epoch

    # model
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.num_encoder_layers = 3
    hparams.num_heads = 8

    tf.logging.info('+++++++++++++++++注意++++++++++++++++')
    tf.logging.info(f'训练数据 {hparams.train_data_dir}')
    tf.logging.info(f'模型保存 {hparams.model_dir}')
    tf.logging.info(f'标签字典 {hparams.labeldict}')
    tf.logging.info('+++++++++++++++++++++++++++++++++++++')

    # infer_from_file('0715/无场景关联1.txt', '0715/无场景关联1_res.txt')
    #
    train()
    # infer()
    # infer_from_file()
    # infer_evaluate('rg_train_20190701_1000002.test_ner',max_count=100000)
    # evaluate()
    # infer_evaluate('test.txt',max_count=10000, out_file='out.txt')
    # infer_evaluate_batch('test.txt',max_count=5000, batch_size=128, out_file='out.txt')
    # infer_evaluate_batch('sampling_test_300.clean.txt',max_count=None, batch_size=128, out_file='test0619_chitsoft.txt')
    # infer_evaluate_batch('testset/0620/sampling_test.clean.txt',max_count=None, batch_size=128, out_file='out/0620/testsoft.txt')
    # for model_id in range(27181, 500000, 27180):
    #     infer_evaluate_batch('testset/0620/sampling_test.clean.txt',max_count=None, batch_size=256, out_file='out/0620/testsig.txt', model_id=model_id)
    # infer_evaluate_batch('testset/0624/0623_ambiscene_点选.clean.txt',max_count=None, batch_size=128, out_file='out/0624/testsoft.txt', model_id='163081')

    # from process_pre import stat_normal, run_strategy, stat_strategy, stat_strategy1
    #
    # # infer_evaluate_batch('testset/0626/first_push.train', max_count=None, batch_size=128, out_file='out/0626/soft/first_push.tsv', model_id='')
    # stat_normal('out/0626/soft/first_push.tsv')
    # run_strategy('out/0626/soft/first_push.tsv', 'out/0626/soft/first_push.strategy.tsv')
    # stat_strategy('out/0626/soft/first_push.strategy.tsv')
    # stat_strategy1('out/0626/soft/first_push.strategy.tsv')

    # import glob
    # from os.path import basename, splitext
    #
    # for file in glob.glob('testset/0626/*.train'):
    #     out_file = 'out/0626/' + splitext(basename(file))[0] + '.tsv'
    #     infer_evaluate_batch(file, max_count=None, batch_size=128, out_file=out_file, model_id='163081')
    #
    #     with open('out/0626/log', 'a', encoding='U8') as f:
    #         f = sys.stdout
    #         stat_normal(out_file, log_file=f)
    #         run_strategy(out_file, splitext(out_file)[0] + '.strategy.tsv')
    #         stat_strategy(splitext(out_file)[0] + '.strategy.tsv', log_file=f)
    #         stat_strategy1(splitext(out_file)[0] + '.strategy.tsv', log_file=f)

    # export_model()
    # test_load_saved_model()
    # test_load_saved_model()
    # test_tfserving_client()
    # stat_class('test0619soft.txt')
