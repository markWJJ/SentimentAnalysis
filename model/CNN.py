import tensorflow as tf
# from model.utils.bimpm import match_utils, layer_utils
# from model.utils.qanet import qanet_layers
# from model.utils.embed import char_embedding_utils
# from loss import point_wise_loss
from model.utils.utils import sent_encoder,self_attention,mean_pool,last_relevant_output,self_attention_topk
from model.model_template import ModelTemplate
from model.utils.transformer_utils import *

class Cnn(ModelTemplate):
    def __init__(self,args,scope):
        super(Cnn, self).__init__(args, scope)
        self.args = args
        self.scope=scope
        self.args.num_layers=1
    def build_emb(self, input, reuse=False,name='',vocab_size=None, *args, **kwargs):
        with tf.variable_scope(name_or_scope="word_embedding_%s"%name, reuse=reuse):
            emb = self.embedding(input,
                                 vocab_size=vocab_size,
                                 num_units=self.args.emb_size,
                                 scale=True,
                                 scope="embed", reuse=reuse)

        return emb

    def conv1d(self,input_tensor, out_channels, filter_size, stride, pool_size, pool_stride, activation, name, reuse):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            conv = tf.layers.conv1d(input_tensor, out_channels, filter_size, strides=stride, padding='valid',
                                    name='conv2')
            conv = activation(conv)
            conv = tf.layers.max_pooling1d(conv, pool_size, pool_stride)
            return conv

    def cnn_encoder(self,input_tensor,name,reuse,all_index,filters):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):

            filter = filters
            res = []

            for index, ele in enumerate(filter):
                with tf.name_scope("conv-maxpool-%s" % index):
                    conv = tf.layers.conv1d(input_tensor, 300, ele, strides=1, padding='valid', name='conv2_ops_%s_%s'%(index,all_index))
                    conv = tf.nn.relu(conv)
                    conv = tf.layers.max_pooling1d(conv, 2, 1)
                    conv,_,_,_=self_attention_topk(conv,scope="conv2_ops_att%s_%s"%(index,all_index),top=1)
                    conv = tf.layers.dropout(conv,self.dropout)
                    res.append(conv)
            ress = tf.stack(res, 1)
            cnn_out = tf.reshape(ress, [-1, self.args.filter_num*len(self.args.filters)])
            # cnn_out = tf.nn.dropout(cnn_out, 0.7)
            # sent_attention=self.intent_attention(self.sent_emb)
            # cnn_out=tf.concat((cnn_out,sent_attention),1)
            return cnn_out


    def build_model(self):
        self.estimation_list=[]
        self.pred_probs_list=[]
        self.logits_list=[]
        self.semantic_losses=[]
        self.args.filters=[3,4,5]
        self.args.filter_num=300
        for i in range(len(self.args.class_nums)):
            with tf.variable_scope(name_or_scope=self.scope + '_enc_%s'%i):
                # sent_token=tf.layers.dropout(self.sent_token,0.1)
                # sent_word_re=tf.layers.dropout(self.sent_word_re,0.1)
                s1_emb = self.build_emb(self.sent_token, vocab_size=self.args.vocab_size,reuse=False,name='emb_%s'%i)
                s1_emb_re = self.build_emb(self.sent_word_re, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)

                s1_emb_char = self.build_emb(self.sent_char, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)
                s1_emb_re_char = self.build_emb(self.sent_word_re_char, vocab_size=self.args.vocab_size,reuse=True,name='emb_%s'%i)

                if i != 0:
                    sent_token_emb=s1_emb_re
                    sent_char_emb=s1_emb_re_char
                    sent_token_len=self.sent_len_re
                    sent_char_len=self.sent_len_re_char
                    # s1_emb=s1_emb_re
                    # sent_len=self.sent_len_re
                    # sent_token=self.sent_word_re
                    # input_mask = tf.sequence_mask(self.sent_len_re, self.seq_len, dtype=tf.float32)
                    # input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1),
                    #                         tf.float32)  # batch_size x seq_len x 1
                else:
                    sent_token_emb = s1_emb_re
                    sent_char_emb = s1_emb_re_char
                    sent_token_len = self.sent_len_re
                    sent_char_len = self.sent_len_re_char
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

                    sent_token_emb = self.cnn_encoder(input_tensor=sent_token_emb, name='conv_%s' % str(i), reuse=False,all_index=i,filters=[3,4,5])
                    s1_flatten_token = tf.layers.flatten(sent_token_emb)

                    sent_char_emb = self.cnn_encoder(input_tensor=sent_char_emb, name='conv_%s_char' % str(i), reuse=False,
                                                  all_index=i,filters=[5,6,7])
                    s1_flatten_char = tf.layers.flatten(sent_char_emb)

                    sent_enc_ = tf.concat([s1_flatten_token, s1_flatten_char], -1)


                    # sent_enc_token=sent_encoder(sent_word_emb=sent_token_emb, hidden_dim=self.args.hidden,
                    #                        sequence_length=sent_token_len, name='sent_enc_token', dropout=self.dropout)
                    # sent_enc_token=tf.layers.dropout(sent_enc_token,self.dropout)
                    # sent_enc_token = mean_pool(sent_enc_token, sent_token_len)
                    #
                    # sent_enc_char=sent_encoder(sent_word_emb=sent_char_emb, hidden_dim=self.args.hidden,
                    #                        sequence_length=sent_char_len, name='sent_enc_char', dropout=self.dropout)
                    # sent_enc_char=tf.layers.dropout(sent_enc_char,self.dropout)
                    # sent_enc_char = mean_pool(sent_enc_char, sent_char_len)
                    #
                    # sent_enc_ = tf.concat([sent_enc_token, sent_enc_char], -1)
                    self.semantic_losses.append(0.0)

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

    def build_model1(self):
        self.estimation_list = []
        self.pred_probs_list = []
        self.logits_list = []

        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        graphs=[g1,g2,g3]

        with g1.as_default():
            s1_emb = self.build_emb(self.sent_token, reuse=False,name='g1')
            input_mask = tf.sequence_mask(self.sent_len, self.seq_len, dtype=tf.float32)
            input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1), tf.float32)  # batch_size x seq_len x 1
            self.build_model_ops(0,s1_emb,self.sent_len,input_mask)
        with g2.as_default():
            input_mask = tf.sequence_mask(self.sent_len_re, self.seq_len, dtype=tf.float32)
            input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1), tf.float32)  # batch_size x seq_len x 1
            s1_emb_re = self.build_emb(self.sent_word_re, reuse=False,name='g2')
            self.build_model_ops(1,s1_emb_re,self.sent_len_re,input_mask)
        with g3.as_default():
            input_mask = tf.sequence_mask(self.sent_len_re, self.seq_len, dtype=tf.float32)
            input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1), tf.float32)  # batch_size x seq_len x 1
            s1_emb_re = self.build_emb(self.sent_word_re, reuse=False,name='g2')
            self.build_model_ops(2,s1_emb_re,self.sent_len_re,input_mask)


    def build_model_ops(self,i,sent_emb,sent_len,input_mask):
        with tf.device('/device:GPU:%s' % self.gpu_id):
            with tf.variable_scope(self.scope + "_enc_%s"%i):
                with tf.variable_scope(name_or_scope=self.scope + '_enc'):
                    with tf.variable_scope(name_or_scope='enc_'):
                        s1_trans_emb = build_transformer_emb(sent_word=self.sent_token, sent_word_emb=sent_emb,
                                                             text_len=self.seq_len,
                                                             args=self.args, dropout=self.dropout, name='left')
                        sen_enc = transformer_encoder(s1_trans_emb, name='self_trans1', args=self.args, dropout=self.dropout,
                                                      reuse=False, context_mask=input_mask,
                                                      num_blocks=self.args.num_blocks)

                        # sen_enc = sent_encoder(sent_word_emb=sent_emb, hidden_dim=self.args.hidden,
                        #                        sequence_length=sent_len, name='sent_enc', dropout=self.dropout)
                        sent_enc_, _ = self_attention(sen_enc, sent_len)

                        # sent_enc_=mean_pool(sen_enc,self.sent_len)
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

