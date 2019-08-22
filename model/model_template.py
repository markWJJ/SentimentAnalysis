import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
import os
from loss import point_wise_loss
gpu_id=5
os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%gpu_id
import logging
from collections import defaultdict
import time
import tqdm
from sklearn.metrics import precision_score, recall_score,f1_score,precision_recall_fscore_support,classification_report
from learning_rate import *
from optimizer import *
from numpy import random
PATH=os.path.split(os.path.realpath(__file__))[0]
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logfile=PATH+'/log/intent.txt'
fh=logging.FileHandler(logfile,mode='w')
fh.setLevel(logging.DEBUG)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter=logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
_logger=logger

class ModelTemplate(object):
    __metaclass__ = ABCMeta
    def __init__(self,args,scope):
        self.args=args
        self.scope=scope
        self.gpu_id=gpu_id
    @abstractmethod
    def build_placeholder(self, ):
        self.seq_len = self.args.seq_len
        self.vocab_size = self.args.vocab_size
        self.emb_size = self.args.emb_size
        self.batch_size = self.args.batch_size

        # ---- place holder -----
        self.sent_token = tf.placeholder(tf.int32, [None, self.seq_len], name='sent1_token')

        self.sent_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent1_len')

        self.sent_poss = tf.placeholder(tf.int32, [None, self.seq_len], name='sent1_poss')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
        self.target = tf.placeholder(shape=(None,), dtype=tf.int32, name='intent')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.grad_clipper=10.0

    @abstractmethod
    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.pred_probs, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.target, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def embedding(self,inputs,vocab_size,num_units,zero_pad=False,scale=True,scope="embedding",reuse=False):
       '''
       :param vocab_size:
       :param num_units:
       :param zero_pad:
       :param scale:
       :param scope:
       :param reuse:
       :return:
       '''
       with tf.variable_scope(scope, reuse=reuse):
           if self.args.use_pre_train_emb:
               assert  vocab_size==self.args.vocab_emb.shape[0]
               lookup_table = tf.get_variable('lookup_table',
                                              dtype=tf.float32,
                                              shape=[vocab_size, num_units],
                                              trainable=False,
                                              initializer=tf.constant_initializer(self.args.vocab_emb, dtype=tf.float32))
           else:
                lookup_table = tf.get_variable('lookup_table',
                                               dtype=tf.float32,
                                               shape=[vocab_size, num_units],
                                               trainable=False,
                                               initializer=tf.contrib.layers.xavier_initializer())
           if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
           outputs = tf.nn.embedding_lookup(lookup_table, inputs)
           if scale:
                outputs = outputs * (num_units ** 0.5)
           self.EmbeddingTable=lookup_table
       return outputs

    def apply_ema(self, *args, **kargs):
        # decay = self.config.get("with_moving_average", None)
        # if decay:
        self.var_ema = tf.train.ExponentialMovingAverage(0.99)
        ema_op = self.var_ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

            self.shadow_vars = []
            self.global_vars = []
            for var in tf.global_variables():
                v = self.var_ema.average(var)
                if v:
                    self.shadow_vars.append(v)
                    self.global_vars.append(var)
            self.assign_vars = []
            for g,v in zip(self.global_vars, self.shadow_vars):
                self.assign_vars.append(tf.assign(g,v))


    @abstractmethod
    def build_loss(self, *args, **kargs):
        with tf.device('/device:GPU:%s' % gpu_id):
            if self.args.loss == "softmax_loss":
                soft_loss, _ = point_wise_loss.softmax_loss(self.estimation, self.target,*args, **kargs)
                self.loss=soft_loss
            elif self.args.loss == "sparse_amsoftmax_loss":
                soft_loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.estimation, self.target, *args, **kargs)
                self.loss=soft_loss
            elif self.args.loss == "focal_loss_binary_v2":

                soft_loss, _ = point_wise_loss.focal_loss_binary_v2(self.estimation, self.target, *args, **kargs)
                self.loss=soft_loss

    @abstractmethod
    def build_op(self):
        with tf.device('/device:GPU:%s' % self.gpu_id):
            if self.args.opt == 'adadelta':
                self.train_op = tf.train.AdadeltaOptimizer(self.args.lr).minimize(self.loss)
            elif self.args.opt == 'adam':
                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = 0.0003  # 初始学习率
                learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                           self.global_step,
                                                           decay_steps=500, decay_rate=0.9)
                self.optimizer = tf.contrib.opt.LazyAdamOptimizer(
                    learning_rate,
                    beta1=0.9,
                    beta2=0.997,
                    epsilon=1e-09)
                trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=trainable_vars)
                params = [var for _, var in grads_and_vars]
                gradients = [grad for grad, _ in grads_and_vars]
                grads, _ = tf.clip_by_global_norm(gradients, self.grad_clipper)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # ADD 2018.06.01
                with tf.control_dependencies(update_ops):  # ADD 2018.06.01
                    self.train_op = self.optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)
                self.saver = tf.train.Saver(max_to_keep=100)
                self.grad_vars = [grad for grad, _ in grads_and_vars]
            elif self.args.opt == 'rmsprop':
                self.train_op = tf.train.RMSPropOptimizer(self.args.lr).minimize(self.loss)

    def iteration(self, sess,epoch, data_loader, train=True):
        """

        :param epoch:
        :param data_loader:
        :param train:
        :return:
        """

        if train:
            flag = 'train'
            batch_num = int(self.args.train_num / self.args.batch_size)
            droput=0.5
        else:
            flag = 'test'
            droput=0.0
            batch_num = int(self.args.test_num / self.args.batch_size)
        sent_word,sent_poss,sent_len,intent=data_loader[:]
        pbar = tqdm.tqdm(total=batch_num)
        avg_acc=0.0
        avg_loss=0.0
        for i in range(batch_num):
            sent_word_,sent_poss_,sent_len_,intent_=sess.run([sent_word,sent_poss,sent_len,intent])
            feed_dict={
                self.sent_token: sent_word_,
                self.sent_poss:sent_poss_,
                self.sent_len: sent_len_,
                self.target:intent_,
                self.dropout:droput
            }
            if train:
                loss,acc,_=sess.run([self.loss,self.accuracy,self.train_op],feed_dict=feed_dict)
            else:
                loss,acc=sess.run([self.loss,self.accuracy],feed_dict=feed_dict)

            avg_loss+=loss
            avg_acc+=acc
            pbar.update(1)

        _logger.info("model_name:%s  EP%d_%s loss=%s acc=%s" % (self.scope,epoch, flag,avg_loss/batch_num,avg_acc/batch_num))
        pbar.close()
        return avg_loss/batch_num,avg_acc/batch_num

    def train(self,train_data_loader,test_data_loader=None,restore_model=None,save_model=None):
        start_time=time.time()
        saver=tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        init_loss=9999
        with tf.Session(config=config) as sess:
            if restore_model:
                saver.restore(sess,restore_model)
                _logger.info("restore model from :%s "%(restore_model))
            else:
                _logger.info("variables_initializer")
                sess.run(tf.global_variables_initializer())
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for epoch in range(self.args.epochs):
                train_loss, train_acc=self.iteration(sess,epoch,train_data_loader,True)
                test_loss, test_acc=self.iteration(sess,epoch,test_data_loader,False)
                _logger.info('\n\n')
                if test_loss< init_loss:
                    self.save(saver,sess,save_model,epoch)
                    init_loss=test_loss
            end_time=time.time()
            coord.request_stop()
            coord.join(threads)
            print('#'*20,end_time-start_time)

    def predict(self,sess,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len):
        start_time=time.time()
        feed_dict = {
            self.sent_token: t1,
            self.sent_len: t1_len,
            self.sent_poss:poss,
            self.sent_word_re:t1_re,
            self.sent_len_re:t1_re_len,
            self.sent_char:t1_char,
            self.sent_len_char:t1_char_len,
            self.sent_word_re_char:t1_rm_char,
            self.sent_len_re_char:t1_rm_char_len,
            self.dropout: 0.0
        }
        # print(sent1_word_)

        pre1,pre2,pre3 = sess.run([self.pred_probs_list[0],self.pred_probs_list[1],self.pred_probs_list[2]], feed_dict=feed_dict)
        res=[]
        for ele in [pre1,pre2,pre3]:
            score=np.max(ele,1)
            pre_leb=np.argmax(ele,1)
            res.append([score,pre_leb])
        return res

    def restore(self,sess,restore_model):
        saver=tf.train.Saver()

        saver.restore(sess, restore_model)
        return sess

    def save(self,saver,sess,save_model,epoch):
        saver.save(sess,save_model,global_step=epoch)
        _logger.info('save in :%s'%(save_model))

