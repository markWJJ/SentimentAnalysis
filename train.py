from model.BaseLstm import BaseLstm
from model.Transformer import BaseLstmStruct
from model.CNN import Cnn
from model.Transformer_cnn import TransformerCNN
from model.LEAM import LEAM
from model.base_transformer_struct import BaseTransformerStruct
from dataset import WordVocab
from dataset.dataset_tfrecord import DataSetTfrecord
import tensorflow as tf
import pickle as pkl
import json
import numpy as np
from urllib import parse
import xlrd
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
import argparse
import os
from numpy import random
from Predict import PredictDataDeal
import jieba.posseg
PATH=os.path.split(os.path.realpath(__file__))[0]
seq_len=50
import logging
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
logger.addHandler(ch)
_logger=logger


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


def build_poss_vocab(train_corpus,test_corpus):
    poss_ele={"None":0}
    index=1
    for path in [train_corpus,test_corpus]:
        for ele in open(path,'r',encoding='utf-8'):
            sent=ele.replace('\n','').split('\t\t')[1]
            sentence_seged = jieba.posseg.cut(sent)
            for x in sentence_seged:
                word, poss = x.word, x.flag
                if poss not in poss_ele:
                    poss_ele[poss]=index
                    index+=1
    return poss_ele

def build_vocab(corpus_path,vocab_path,mode):
    with open(corpus_path, "r",encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=None, min_freq=1,mode=mode)
        print("VOCAB SIZE:", len(vocab))
        vocab.save_vocab(vocab_path)


def build_label_vocab(train_corpus,test_corpus):
    intent_dict={"None":0}
    intent_index=len(intent_dict)
    for path in [train_corpus,test_corpus]:
        for ele in open(path,'r',encoding='utf-8'):
            eles=ele.replace('\n','').split('\t\t')
            intent=eles[0]
            if intent not  in intent_dict:
                intent_dict[intent]=intent_index
                intent_index+=1
    return intent_dict


def read_tfRecord(seq_len,batch_size):
    def _read_tfrecoder(file_tfRecord):

        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(file_tfRecord),
            shuffle=True, num_epochs=None)  # None表示没哟限制
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
                serialized_example,
                features={
                'sent_word': tf.FixedLenFeature([seq_len], tf.int64),
                'poss': tf.FixedLenFeature([seq_len], tf.int64),
                'sent_len':tf.FixedLenFeature([1], tf.int64),
                'intent': tf.FixedLenFeature([1], tf.int64)

                }
                )
        sent_word = tf.cast(features['sent_word'], tf.int32)
        sent_poss = tf.cast(features['poss'], tf.int32)
        sent_len = tf.cast(features['sent_len'], tf.int32)
        intent= tf.cast(features['intent'], tf.int32)
        sent_word=tf.reshape(sent_word,[-1,seq_len])
        sent_poss=tf.reshape(sent_poss,[-1,seq_len])
        input_queue = tf.train.slice_input_producer([sent_word,sent_poss,sent_len,intent], shuffle=True)
        sent_word_batch,sent_poss_batch, sent_len_batch,intent_batch \
            = tf.train.batch(input_queue,batch_size=batch_size,allow_smaller_final_batch=True,num_threads=4)

        return [sent_word_batch,sent_poss_batch, sent_len_batch,intent_batch]
    return _read_tfrecoder

class PreTrainVocab(object):

    def __init__(self,pre_train_path,emb_size):
        self.unkow_dict={}
        self.emb_size=emb_size
        self.w2v_model=pkl.load(open(pre_train_path,'rb'))

    def embedding_api(self,word):
        word = parse.quote(word)
        try:
            vec = self.w2v_model[word]
        except:
            if word not in self.unkow_dict:
                self.unkow_dict[word] = random.rand((self.emb_size))
            vec = self.unkow_dict[word]
        return vec

    def getEmbeddimhArray(self,vocab):
        emb_list=[]
        for k,v in vocab.stoi.items():
            emb=self.embedding_api(k)
            emb_list.append(emb)
        return np.array(emb_list,dtype=np.float32)

def train(args,model_name):

    if not os.path.exists(PATH+args.output_path):
        os.mkdir(PATH+args.output_path)

    _logger.info("new_vocab:%s"%args.new_vocab)
    _logger.info("use_tfrecord:%s"%args.use_tfrecord)
    _logger.info("train_dataset:%s"%args.train_dataset)
    _logger.info("test_dataset:%s"%args.test_dataset)
    _logger.info("task_name:%s"%args.task_name)
    _logger.info("model_name:%s"%args.model_name)
    _logger.info("new_tfrecord:%s"%args.new_tfrecord)
    _logger.info("restore_model:%s"%args.restore_model)
    _logger.info("use_pre_train_emb:%s"%args.use_pre_train_emb)


    _logger.info("build label vocab")
    if args.new_label_vocab:
        intent_dict = build_label_vocab(args.train_dataset, args.test_dataset)
        _logger.info("%s %s"%(intent_dict,len(intent_dict)))
        pkl.dump(intent_dict, open(PATH + "/intent_dict.p", 'wb'))

    intent_dict = pkl.load(open(PATH + "/intent_dict.p", 'rb'))
    args.id2intent={v:k for k,v in intent_dict.items()}

    ### load word_vocab
    if not args.new_vocab and os.path.exists(PATH+args.vocab_path):
        _logger.info("Loading Vocab: %s"%(PATH+args.vocab_path))
        vocab = WordVocab.load_vocab(PATH+args.vocab_path)
    else:
        _logger.info("build vocab")
        build_vocab(args.train_dataset,PATH+args.vocab_path,mode='word_char')
        _logger.info("Loading Vocab: %s" % (PATH+args.vocab_path))
        vocab = WordVocab.load_vocab(PATH+args.vocab_path)

    _logger.info("Vocab Size:%s"%(len(vocab)))

    poss_vocab = build_poss_vocab(args.train_dataset, args.test_dataset)
    pkl.dump(poss_vocab, open(PATH + "/poss_vocab.p", 'wb'))
    poss_vocab = pkl.load(open(PATH + "/poss_vocab.p", 'rb'))

    ### load pre_train Embedding
    # print(vocab.stoi)
    args.num_layers=1
    args.vocab_size=len(vocab)
    args.class_nums = len(intent_dict)
    args.poss_num=len(poss_vocab)

    if args.use_pre_train_emb:
        if args.new_pre_vocab:
            pre_emb_cls = PreTrainVocab(args.pre_train_emb_path, args.pre_train_emb_size)
            vocab_emb=pre_emb_cls.getEmbeddimhArray(vocab)
            pkl.dump(vocab_emb,open('%s_vocab_emb.p'%args.task_name,'wb'))
            args.vocab_emb = vocab_emb
        else:
            vocab_emb=pkl.load(open('%s_vocab_emb.p'%args.task_name,'rb'))
            args.vocab_emb = vocab_emb
        _logger.info('load pre_train_emb finish emb_array size:%s'%(len(vocab_emb)))

    ### build tfrecord
    if not os.path.exists(PATH+args.train_tfrecord_path) or not os.path.exists(PATH+args.test_tfrecord_path) or args.new_tfrecord:
        _logger.info('building tfrecords')
        DataSetTfrecord(args.train_dataset, vocab, args.seq_len,intent_dict=intent_dict,poss_vocab=poss_vocab,out_path=PATH+args.train_tfrecord_path,)
        DataSetTfrecord(args.test_dataset, vocab, args.seq_len,poss_vocab=poss_vocab, intent_dict=intent_dict, out_path=PATH+args.test_tfrecord_path)
    _read_tfRecord=read_tfRecord(args.seq_len,args.batch_size)
    _logger.info("loading tfrecords")
    train_data_loader=_read_tfRecord(PATH+args.train_tfrecord_path)
    test_data_loader=_read_tfRecord(PATH+args.test_tfrecord_path)
    train_num=int([e for e in open(PATH+args.train_tfrecord_path+".index",'r',encoding='utf-8').readlines()][0])
    test_num=int([e for e in open(PATH+args.test_tfrecord_path+".index",'r',encoding='utf-8').readlines()][0])
    _logger.info('train_num:%s  test_num:%s'%(train_num,test_num))
    args.train_num=train_num
    args.test_num=test_num


    _logger.info('%s  batch_size:%s  use_tfrecod:%s'%(args.model_name,args.batch_size,args.use_tfrecord))
    for index,e in enumerate(train_data_loader):
        if index%10:
            print(e)

    # ### 模型选择 BaseTransformerStruct
    # model_name=args.model_name
    if model_name == 'BaseLSTM':
        model=BaseLstm(args,'BaseLstm')
    elif model_name == 'BaseLstmStruct':
        model=BaseLstmStruct(args,'BaseLstmStruct')
    elif model_name == 'BaseTransformerStruct':
        model = BaseTransformerStruct(args, 'BaseTransformerStruct')
    elif model_name == 'cnn':
        model = Cnn(args, 'cnn')
    elif model_name == 'TransformerCNN':
        model = TransformerCNN(args, 'TransformerCNN')
    elif model_name == 'LEAM':
        model = LEAM(args, 'LEAM')
    args.model_name=model_name
    model.build_placeholder()
    model.build_model()
    model.build_accuracy()
    model.build_loss()
    model.build_op()

    if args.restore_model=='':
        model.train(train_data_loader, test_data_loader,restore_model=None, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))
    else:
        model.train(train_data_loader, test_data_loader,restore_model=PATH+args.restore_model, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))

def predict(args,model_name,restore_path):
    intent_dict = pkl.load(open(PATH + "/intent_dict.p", 'rb'))
    args.id2intent = {v: k for k, v in intent_dict.items()}
    vocab = WordVocab.load_vocab(PATH + args.vocab_path)
    poss_vocab = pkl.load(open(PATH + "/poss_vocab.p", 'rb'))

    args.num_layers = 1
    args.vocab_size = len(vocab)
    args.class_nums = len(intent_dict)
    args.poss_num = len(poss_vocab)

    if args.use_pre_train_emb:
        vocab_emb = pkl.load(open('%s_vocab_emb.p' % args.task_name, 'rb'))
        args.vocab_emb = vocab_emb

    if model_name == 'BaseLSTM':
        model = BaseLstm(args, 'BaseLstm')
    elif model_name == 'BaseLstmStruct':
        model = BaseLstmStruct(args, 'BaseLstmStruct')
    elif model_name == 'BaseTransformerStruct':
        model = BaseTransformerStruct(args, 'BaseTransformerStruct')
    elif model_name == 'cnn':
        model = Cnn(args, 'cnn')
    elif model_name == 'TransformerCNN':
        model = TransformerCNN(args, 'TransformerCNN')
    elif model_name == 'LEAM':
        model = LEAM(args, 'LEAM')
    args.model_name = model_name
    model.build_placeholder()
    model.build_model()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess=model.restore(sess,restore_path)
        pdd=PredictDataDeal(vocab=vocab,seq_len=args.seq_len,poss_vocab=poss_vocab,vocab_char=None)
        while True:
            sent=input("输入:")
            t1,t1_len,poss=pdd.predict(sent)
            pre_prob,pre_label=model.predict(sess,t1,t1_len,poss)
            print(args.id2intent[pre_label[0]],np.max(pre_prob,-1))


def main(argsConfig):
    model_name='BaseLSTM'
    args_dict = json.load(open(argsConfig.config_path, 'r'))
    args = dict_to_object(dict(args_dict))

    if argsConfig.mode=='train':
        train(args,model_name)


    elif argsConfig.mode == 'predict':
        restore_path = PATH + "/output/%s_%s_2kw.ckpt" % (model_name, args.task_name)
        predict(args,model_name,restore_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="mode", type=str, default='')  # vocab_8kw SeqLstm
    parser.add_argument("--config_path",
                        help="config_path", type=str, default=PATH + '/Configs/BaseLstm.config')  # vocab_8kw
    argsConfig = parser.parse_args()

    main(argsConfig)