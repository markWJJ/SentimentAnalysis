import tensorflow as tf
# from model.utils.bimpm import match_utils, layer_utils
# from model.utils.qanet import qanet_layers
# from model.utils.embed import char_embedding_utils
# from loss import point_wise_loss
from model.utils.utils import sent_encoder,self_attention,mean_pool,last_relevant_output
from model.model_template import ModelTemplate
from model.utils.transformer_utils import *

class BaseLstmStruct(ModelTemplate):
    def __init__(self,args,scope):
        super(BaseLstmStruct, self).__init__(args, scope)
        self.args = args
        self.scope=scope
        self.args.num_layers=1

    def conv1d(self,input_tensor, out_channels, filter_size, stride, pool_size, pool_stride, activation, name, reuse):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            conv = tf.layers.conv1d(input_tensor, out_channels, filter_size, strides=stride, padding='valid',
                                    name='conv2')
            conv = activation(conv)
            conv = tf.layers.max_pooling1d(conv, pool_size, pool_stride)
            return conv

    def cnn_encoder(self,input_tensor,name,reuse,all_index):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):

            filter = self.args.filters
            res = []

            for index, ele in enumerate(filter):
                with tf.name_scope("conv-maxpool-%s" % index):
                    conv = tf.layers.conv1d(input_tensor, 300, ele, strides=1, padding='valid', name='conv2_ops_%s_%s'%(index,all_index))
                    conv = tf.nn.relu(conv)
                    conv = tf.layers.max_pooling1d(conv, 2, 1)
                    conv,_=self_attention(conv,scope="conv2_ops_att%s_%s"%(index,all_index))
                    conv = tf.layers.dropout(conv,self.dropout)
                    res.append(conv)
            ress = tf.stack(res, 1)
            cnn_out = tf.reshape(ress, [-1, self.args.filter_num*len(self.args.filters)])
            # cnn_out = tf.nn.dropout(cnn_out, 0.7)
            # sent_attention=self.intent_attention(self.sent_emb)
            # cnn_out=tf.concat((cnn_out,sent_attention),1)
            return cnn_out

    def build_emb(self, sent_word, vocab_size=None,reuse=False,name=None, *args, **kwargs):
        with tf.variable_scope(name_or_scope="word_embedding_%s"%name, reuse=reuse):
            encoder_input_emb = self.embedding(sent_word,
                                               vocab_size=vocab_size,
                                               num_units=self.args.emb_size,
                                               scale=True,
                                               scope="embed", reuse=reuse)

            if self.args.sinusoid:
                encoder_input_emb += positional_encoding(sent_word,
                                                         num_units=vocab_size,
                                                         zero_pad=False,
                                                         scale=False,
                                                         scope="enc_pe", reuse=reuse)
            else:
                encoder_input_emb += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(sent_word)[1]), 0),
                            [tf.shape(sent_word)[0], 1]),
                    vocab_size=self.args.seq_len,
                    num_units=self.args.emb_size,
                    zero_pad=False,
                    scale=False,
                    scope="enc_pe", reuse=reuse)

            ## Dropout
            encoder_input_emb = tf.layers.dropout(encoder_input_emb,
                                                  rate=self.dropout,
                                                  training=True)

            return encoder_input_emb

    def transformer_encoder(self, encoder_emb1, encoder_emb2, name, reuse=False, num_blocks=4):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            data = []
            enc = encoder_emb1
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              num_units=self.args.emb_size,
                                              num_heads=self.args.num_heads,
                                              dropout_rate=self.dropout,
                                              is_training=True,
                                              causality=False)

                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4 * self.args.emb_size, self.args.emb_size])
                    data.append(enc)
            return data[-1]

    def build_model(self):

        with tf.variable_scope(name_or_scope=self.scope + '_enc'):
            s1_emb = self.build_emb(self.sent_token, vocab_size=self.args.vocab_size,reuse=False,name='emb_%s'%i)
            s1_emb_re = self.build_emb(self.sent_word_re, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)

            s1_emb_char = self.build_emb(self.sent_char, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)
            s1_emb_re_char = self.build_emb(self.sent_word_re_char, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)


            s1_emb_neg = self.build_emb(self.sent_token_neg, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)
            s1_emb_char_neg = self.build_emb(self.sent_char_neg, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)

            if i != 0:
                sent_token_emb=s1_emb
                sent_char_emb=s1_emb_char
                sent_token_len=self.sent_len
                sent_char_len=self.sent_len_char
                # s1_emb=s1_emb_re
                # sent_len=self.sent_len_re
                # sent_token=self.sent_word_re
                # input_mask = tf.sequence_mask(self.sent_len_re, self.seq_len, dtype=tf.float32)
                # input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1),
                #                         tf.float32)  # batch_size x seq_len x 1
            else:
                sent_token_emb=s1_emb
                sent_char_emb=s1_emb_char
                sent_token_len=self.sent_len
                sent_char_len=self.sent_len_char
                # sent_token_emb=s1_emb
                # sent_char_emb=s1_emb_char
                # sent_token_len=self.sent_len
                # sent_char_len=self.sent_len_char
                # sent_len=self.sent_len
                # sent_token=self.sent_token
                # input_mask = tf.sequence_mask(self.sent_len, self.seq_len, dtype=tf.float32)
                # input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1),
                #                      tf.float32)  # batch_size x seq_len x 1

            with tf.variable_scope(name_or_scope='_%s'%i):

                sent_enc_token = self.transformer_encoder(sent_token_emb, sent_token_emb, name='self_trans', reuse=False, num_blocks=2)
                # sent_enc_token=tf.multiply(sent_enc_token,tf.expand_dims(self.key_emb,2))
                # sent_enc_token = mean_pool(sent_enc_token, sent_token_len)
                sent_enc_token,_ = self_attention(sent_enc_token,sent_token_len,scope='s_0')
                # sent_enc_token,_ = self_attention(sent_enc_token,sent_token_len,scope='s_0')

                sent_enc_char = self.transformer_encoder(sent_char_emb, sent_char_emb, name='self_trans_char', reuse=False, num_blocks=2)
                sent_enc_char = mean_pool(sent_enc_char, sent_char_len)
                sent_enc_char,_ = self_attention(sent_enc_char,sent_char_len,scope='s_1')
                sent_enc_ = tf.concat([sent_enc_token, sent_enc_char], -1)
                # sent_enc_ = sent_enc_token



                s1_emb_neg = self.transformer_encoder(s1_emb_neg, s1_emb_neg, name='self_trans', reuse=True, num_blocks=2)
                # s1_emb_neg = mean_pool(s1_emb_neg, self.sent_len_neg)
                s1_emb_neg,_ = self_attention(s1_emb_neg, self.sent_len_neg,scope='s_0',reuse=True)

                s1_emb_char_neg = self.transformer_encoder(s1_emb_char_neg, s1_emb_char_neg, name='self_trans_char', reuse=True, num_blocks=2)
                # s1_emb_char_neg = mean_pool(s1_emb_char_neg, self.sent_char_len_neg,)
                s1_emb_char_neg,_ = self_attention(s1_emb_char_neg, self.sent_char_len_neg,scope='s_1',reuse=True)

                sent_enc_neg = tf.concat([s1_emb_neg, s1_emb_char_neg], -1)
                s1_emb_enc,s2_emb_enc=sent_enc_,sent_enc_neg
                query_norm = tf.sqrt(tf.reduce_sum(tf.square(s1_emb_enc), 1, True))
                doc_norm = tf.sqrt(tf.reduce_sum(tf.square(s2_emb_enc), 1, True))
                prod = tf.reduce_sum(tf.multiply(s1_emb_enc, s2_emb_enc), 1, True)
                norm_prod = tf.multiply(query_norm, doc_norm) + 0.01
                cos_sim = tf.truediv(prod, norm_prod)
                neg_cos_sim = tf.abs(1 - cos_sim)
                with tf.variable_scope(name_or_scope='semantic_out'):
                    estimation_semantic = tf.concat([neg_cos_sim, cos_sim], 1)
                semantic_traget=tf.zeros_like(self.target1)
                semantic_loss = tf.losses.sparse_softmax_cross_entropy(labels=semantic_traget, logits=estimation_semantic)
                self.semantic_losses.append(semantic_loss)
                # self.semantic_losses.extend(tf.reduce_mean(tf.zeros_like(self.target1)))
                # sent_enc_, _ = self_attention(sen_enc, sent_len)

                #
                # sent_enc_=tf.reshape(tf.cast([sent_enc_,s1_flatten],1),[-1,1200])
                # self.estimation=tf.layers.dense(sent_enc_,self.args.class_num)
                estimation = tf.contrib.layers.fully_connected(
                    inputs=sent_enc_,
                    num_outputs=self.args.class_nums[i],
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC"
                )
                pred_probs = tf.contrib.layers.softmax(estimation)
                logits = tf.cast(tf.argmax(pred_probs, -1), tf.int32)

                self.estimation_list.append(estimation)
                self.pred_probs_list.append(pred_probs)
                self.logits_list.append(logits)

