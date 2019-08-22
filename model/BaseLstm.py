import tensorflow as tf
# from model.utils.bimpm import match_utils, layer_utils
# from model.utils.qanet import qanet_layers
# from model.utils.embed import char_embedding_utils
# from loss import point_wise_loss
from model.utils.utils import sent_encoder,self_attention,mean_pool,last_relevant_output
from model.model_template import ModelTemplate
from model.utils.transformer_utils import *

class BaseLstm(ModelTemplate):
    def __init__(self,args,scope):
        super(BaseLstm, self).__init__(args, scope)
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


    def build_model(self):
        self.args.filters=[3,4,5]
        self.args.filter_num=300
        with tf.variable_scope(name_or_scope=self.scope + '_enc'):
            s1_emb = self.build_emb(self.sent_token, vocab_size=self.args.vocab_size,reuse=False,name='emb')
            with tf.variable_scope(name_or_scope='encoder'):
                sent_enc_token=sent_encoder(sent_word_emb=s1_emb, hidden_dim=self.args.hidden,
                                       sequence_length=self.sent_len, name='sent_enc_token', dropout=self.dropout)
                sent_enc_token=tf.layers.dropout(sent_enc_token,self.dropout)
                sent_enc_token = mean_pool(sent_enc_token, self.sent_len)
                self.estimation = tf.contrib.layers.fully_connected(
                    inputs=sent_enc_token,
                    num_outputs=self.args.class_nums,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC"
                )
                self.pred_probs = tf.contrib.layers.softmax(self.estimation)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)


