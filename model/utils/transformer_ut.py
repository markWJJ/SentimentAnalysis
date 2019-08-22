import tensorflow as tf
import numpy as np
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

