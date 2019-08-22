import tensorflow as tf
# from model.utils.bimpm import match_utils, layer_utils
# from model.utils.qanet import qanet_layers
# from model.utils.embed import char_embedding_utils
# from loss import point_wise_loss
from model.utils.utils import sent_encoder,self_attention,mean_pool,last_relevant_output
from model.model_template import ModelTemplate
from model.utils.transformer_utils import *
# from model.utils.self_attn import *
class BaseTransformerStruct(ModelTemplate):
    def __init__(self,args,scope):
        super(BaseTransformerStruct, self).__init__(args, scope)
        self.args = args
        self.scope=scope
        self.args.num_layers=1
    def build_emb(self, input,emb_size,name="word_embedding", emb_dim=300,reuse=False, *args, **kwargs):
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            emb = self.embedding(input,
                                 vocab_size=emb_size,
                                 num_units=emb_dim,
                                 scale=True,
                                 scope="embed", reuse=reuse)

        return emb


    def build_model(self):
        with tf.device('/device:GPU:%s' % self.gpu_id):

            s1_emb = self.build_emb(self.sent_token,emb_size=self.args.vocab_size,emb_dim=300,name='word_emb',reuse=False)
            s1_emb_seg = self.build_emb(self.sent_seg,emb_size=self.args.seg_num,emb_dim=300,name='word_emb_seg',reuse=False)
            s1_emb_poss = self.build_emb(self.sent_poss,emb_size=self.args.poss_num,emb_dim=300,name='word_emb_poss',reuse=False)

            input_mask = tf.sequence_mask(self.sent_len,self.seq_len)
            input_mask = tf.cast(tf.expand_dims(input_mask, axis=-1),tf.float32)  # batch_size x seq_len x 1

            # s1_emb *= input_mask
            # s1_emb_seg *= input_mask
            # s1_emb_poss *= input_mask

            self.args.proximity_bias=False
            self.args.pos='emb'
            self.args.layer_prepostprocess_dropout=0.1
            self.args.num_encoder_layers=1
            self.args.hidden_size=self.args.emb_size
            self.args.num_heads=self.args.num_heads
            self.args.attention_dropout=0.1
            self.args.self_attention_type="dot_product"
            self.args.max_relative_position=5
            self.args.max_length=self.seq_len
            self.args.attention_variables_3d=False
            self.args.layer_postprocess_sequence="da"  #dropout add_previous Normal
            self.args.layer_preprocess_sequence="n"
            self.args.activation_dtype="bfloat32"
            self.args.use_target_space_embedding=False
            self.args.attention_dropout_broadcast_dims=""
            self.args.use_pad_remover=False
            self.args.norm_type='layer'
            self.args.norm_epsilon=1e-6
            self.args.layer_prepostprocess_dropout_broadcast_dims=''
            self.args.attention_key_channels=0
            self.args.attention_value_channels=0
            self.args.relu_dropout=0.2
            self.args.conv_first_kernel=3
            self.args.ffn_layer="dense_relu_dense"
            self.args.relu_dropout_broadcast_dims=''
            self.args.filter_size=512
            self.args.weight_decay=1e-5,



            # emb=tf.concat([s1_emb,s1_emb_seg,s1_emb_poss],2)
            emb=s1_emb
            # emb+=s1_emb_seg
            # emb+=s1_emb_poss
            with tf.variable_scope(name_or_scope=self.scope+'_enc'):
                encoder_output = transformer_encoder_ht(emb,target_space=None,hparams=self.args,features=None,losses=None)

                # input_mask = tf.squeeze(input_mask, axis=-1)
                # v_attn = multi_dimensional_attention(
                #     encoder_output, input_mask, 'multi_dim_attn_for_%s' % "atten",
                #                                 1 - self.dropout, True, self.args.weight_decay, "relu")
                #
                # v_sum = tf.reduce_sum(encoder_output, 1)
                # v_ave = tf.div(v_sum, input_mask)
                # v_max = tf.reduce_max(encoder_output, 1)
                #
                # out = tf.concat([v_ave, v_max, v_attn], axis=-1)


                print("#####encoder_output",encoder_output)
                sen_enc=encoder_output
                # print("context_mask",input_mask)
                # dropout_rate = tf.cond(is_training,
                #                        lambda: config.dropout_rate,
                #                        lambda: 0.0)
                s1_trans_emb = build_transformer_emb(sent_word=self.sent_token, sent_word_emb=emb, text_len=self.seq_len,
                                                     args=self.args, dropout=0.0, name='left')
                sen_enc = transformer_encoder(s1_trans_emb, name='self_trans1', args=self.args, dropout=0.0,
                                              reuse=False,context_mask=input_mask, num_blocks=self.args.num_blocks)

                ner_sent_emb=tf.layers.dense(sen_enc,self.args.ner_num)
                self.ner_soft=tf.nn.softmax(ner_sent_emb,-1)
                # ner_pre_label=tf.argmax(self.ner_soft,-1)
                # self.ner_loss=tf.losses.sparse_softmax_cross_entropy(self.sent_ner,self.ner_soft)
                # self.ner_pre_label=ner_pre_label
                # ner_struct=ner_pre_label

                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    ner_sent_emb, self.sent_ner, self.sent_len)
                self.trans_params = trans_params  # need to evaluate it for decoding

                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                    ner_sent_emb, trans_params, self.sent_len)
                ner_pre_label=viterbi_sequence
                correct = tf.equal(
                    tf.cast(ner_pre_label, tf.int32),
                    tf.cast(self.sent_ner, tf.int32)
                )
                self.ner_pre_label=viterbi_sequence
                ner_struct=viterbi_sequence
                self.ner_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                self.ner_loss = tf.reduce_mean(-log_likelihood)
                ner_struct_mask=tf.cast(tf.expand_dims(ner_struct,2),tf.float32)
                print('ner_struct',ner_struct)




                # sen_attention=self_attention(sen_enc,sequence_len)
                #
                # if self.args.GetEmbtype == 'mean_pool':
                #     sent_enc_=mean_pool(sen_enc,self.sent_len)
                # else:
                #     sent_enc_=last_relevant_output(sen_enc,self.sent_len)
                # sent_enc_=tf.concat([sent_enc_,sen_attention],1)
                # sent_enc_=last_relevant_output(sen_enc,self.sent_len)
                sent_enc_,_=self_attention(sen_enc,self.sent_len)

                # sent_enc_=tf.multiply(sen_enc,ner_struct_mask)
                sent_enc_=tf.multiply(sen_enc,tf.cast(tf.expand_dims(self.sent_ner,2),tf.float32))
                sent_enc_ = tf.reduce_mean(sent_enc_, 1)

                # sent_enc_=mean_pool(sen_enc,self.sent_len)
                # self.estimation=tf.layers.dense(sent_enc_,self.args.class_num)
                self.estimation = tf.contrib.layers.fully_connected(
                    inputs=sent_enc_,
                    num_outputs=self.args.class_num,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.args.l2_reg),
                    biases_initializer=tf.constant_initializer(1e-04),
                    scope="FC"
                )
                self.l2_loss= 0.0
                self.pred_probs = tf.contrib.layers.softmax(self.estimation)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)


