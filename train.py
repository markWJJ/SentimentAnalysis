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

def train(argsConfig):
    args_dict=json.load(open(argsConfig.config_path,'r'))
    args=dict_to_object(dict(args_dict))
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
    model_name="BaseLSTM"
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



    do_train=True
    do_predict=False
    do_predict_txt=False
    do_predict_other=False
    do_predict_xlsx=False
    print(argsConfig.mode)
    if argsConfig.mode=='train':
        do_train=True
    if argsConfig.mode=='predict':
        do_predict=True
    if argsConfig.mode=='predict_txt':
        do_predict_txt=True
    if argsConfig.mode=='predict_other':
        do_predict_other=True
    if argsConfig.mode=='predict_xlsx':
        do_predict_xlsx=True

    restore_path=PATH + "/output/%s_%s_2kw.ckpt-15"%(model_name,args.task_name)
    if do_train:
        if args.restore_model=='':
            model.train(train_data_loader, test_data_loader,restore_model=None, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))
        else:
            model.train(train_data_loader, test_data_loader,restore_model=PATH+args.restore_model, save_model=PATH + "/output/%s_%s_2kw.ckpt"%(model_name,args.task_name))

    # intent2name = pkl.load(open('./intent2name.p', 'rb'))

    if do_predict:
        config = tf.ConfigProto(allow_soft_placement=True)
        id2label1 = {v: k for k, v in label_vocab1.items()}
        id2label2 = {v: k for k, v in label_vocab2.items()}
        id2label3 = {v: k for k, v in label_vocab3.items()}
        id2label_list=[id2label1,id2label2,id2label3]

        with tf.Session(config=config) as sess:
            sess=model.restore(sess,restore_path)
            pdd=PredictDataDeal(vocab=vocab,seq_len=args.seq_len,poss_vocab=poss_vocab,vocab_char=None)
            # id2label={v:k for k,v in label_vocab.items()}
            # id2ner={v:k for k,v in ner_vocab.items()}
            while True:
                ss=[]
                sent=input("输入:")
                t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len=pdd.predict(sent)
                result=model.predict(sess,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len)
                for i,res in enumerate(result):
                    id2label=id2label_list[i]
                    # print(res[0][0])
                    # print(id2label[res[1][0]])
                    ss.append("_".join([str(res[0][0]),str(id2label[res[1][0]])]))
                print(ss)

    if do_predict_txt:
        config = tf.ConfigProto(allow_soft_placement=True)
        path='/dockerdata/KeyWordDataSet/rg_train_20190701_1000002.test_new_0723'
        id2label1 = {v: k for k, v in label_vocab1.items()}
        id2label2 = {v: k for k, v in label_vocab2.items()}
        id2label3 = {v: k for k, v in label_vocab3.items()}
        id2label_list=[id2label1,id2label2,id2label3]
        true_label_list=[]
        pre_label_list=[]
        fw2=open('out_analy_label2.txt','w',encoding='utf-8')
        fw3=open('out_analy_label3.txt','w',encoding='utf-8')
        with tf.Session(config=config) as sess:
            sess = model.restore(sess,restore_path)
            pdd = PredictDataDeal(vocab=vocab, seq_len=args.seq_len, poss_vocab=poss_vocab)

            for line in open(path,'r',encoding='utf-8').readlines():
                lines=line.replace('\n','').split(":")
                sent,labels=lines[-2],lines[-1]
                label_list=labels.split('_')
                true_label_list.append(label_list)
                t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len=pdd.predict(sent)
                result=model.predict(sess,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len)
                ss=[]
                for i, res in enumerate(result):
                    id2label = id2label_list[i]
                    # print(res[0][0])
                    # print(id2label[res[1][0]])
                    ss.append(id2label[res[1][0]])

                if ss[1]!=label_list[1]:
                    fw2.write(sent+'\t\t'+str(label_list[1])+'\t\t'+str(ss[1])+'\n')
                if ss[2]!=label_list[2]:
                    fw3.write(sent+'\t\t'+str(label_list[2])+'\t\t'+str(ss[2])+'\n')
                pre_label_list.append(ss)

        true_label1=[e[0] for e in true_label_list]
        true_label2=[e[1] for e in true_label_list]
        true_label3=[e[2] for e in true_label_list]

        pre_label1=[e[0] for e in pre_label_list]
        pre_label2=[e[1] for e in pre_label_list]
        pre_label3=[e[2] for e in pre_label_list]

        report1=classification_report(true_label1,pre_label1)
        report2=classification_report(true_label2,pre_label2)
        report3=classification_report(true_label3,pre_label3)

        print(report1)
        print(report2)
        print(report3)
        labels3=[]
        for e in true_label3:
            if e not in labels3:
                labels3.append(e)

        labels2=[]
        for e in true_label2:
            if e not in labels2:
                labels2.append(e)
        print(labels3)
        matrix_2=confusion_matrix(true_label2, pre_label2, labels=labels2, sample_weight=None)
        matrix_3=confusion_matrix(true_label3, pre_label3, labels=labels3, sample_weight=None)

        fw_3=open('confusion_matrix_label3.txt','w',encoding='utf-8')
        for i in range(len(matrix_3)):
            ll=labels3[i]
            ele=matrix_3[i]
            fw_3.write(ll+':'+'\t\t')
            for j,e in enumerate(ele):
                if int(e)>0:
                    fw_3.write(labels3[j]+'_'+str(e)+'\t\t')
            fw_3.write('\n')

        fw_2=open('confusion_matrix_label2.txt','w',encoding='utf-8')
        for i in range(len(matrix_2)):
            ll=labels2[i]
            ele=matrix_2[i]
            fw_2.write(ll+':'+'\t\t')
            for j,e in enumerate(ele):
                if int(e)>0:
                    fw_2.write(labels2[j]+'_'+str(e)+'\t\t')
            fw_2.write('\n')





    if do_predict_xlsx:

        # ic=IntentCls(model_name=restore_path)
        dd_2={}
        for ele in open('./test_with_c1_02.txt','r',encoding='utf-8').readlines():
            eles=ele.replace('\n','').split('\t\t')
            sent=eles[3]
            score=eles[2]
            label=eles[1]
            if label.__contains__('inter') or label.__contains__('自动续费'):
                label="None"
            dd_2[sent]=[label,score]

        dd_3={}
        for ele in open('./test_03.txt','r',encoding='utf-8').readlines():
            eles=ele.replace('\n','').split('\t\t')
            sent=eles[3]
            score=eles[2]
            label=eles[1]
            if label.__contains__('inter') or label.__contains__('自动续费'):
                label="None"
            dd_3[sent]=[label,score]

        # dd_2={}
        # dd_3={}
        config = tf.ConfigProto(allow_soft_placement=True)
        path='./cp_4000142_0726.xlsx'
        id2label1 = {v: k for k, v in label_vocab1.items()}
        id2label2 = {v: k for k, v in label_vocab2.items()}
        id2label3 = {v: k for k, v in label_vocab3.items()}
        id2label_list=[id2label1,id2label2,id2label3]
        true_label_list=[]
        pre_label_list=[]
        fw=open("cp_4000142_0723_out.txt",'w',encoding='utf-8')
        fw_false=open("cp_4000142_0723_out_false.txt",'w',encoding='utf-8')

        count_dict={}
        with tf.Session(config=config) as sess:
            sess = model.restore(sess, restore_path)
            pdd = PredictDataDeal(vocab=vocab, seq_len=args.seq_len, poss_vocab=poss_vocab)


            work_sheet=xlrd.open_workbook(path)
            sheet=work_sheet.sheet_by_index(1)
            stand_list = ["自动续费_撤销_None","自动续费_None_None","自动续费_退款_None","自动续费_开通_None","自动续费_查询_None"]
            for i in range(1,sheet.nrows):
                sent=sheet.cell_value(i,2)
                label=sheet.cell_value(i,3)
                entity=entity_detect(sent)
                flag=True
                t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len=pdd.predict(sent)
                result=model.predict(sess,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len)
                # result=ic.predict(t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len)
                ss=[]
                pp_ss=[]
                for i, res in enumerate(result):
                    id2label = id2label_list[i]
                    # print(res[0][0])
                    # print(id2label[res[1][0]])
                    if i==1:
                        if sent in dd_2:
                            ss.append(str(dd_2[sent][1])+'_'+dd_2[sent][0])
                            pp_ss.append(dd_2[sent][0])
                        else:
                            if res[0][0] <= 0.5:
                                ss.append("0.0" + '_' + 'None')
                                pp_ss.append('None')
                            else:
                                ss.append(str(res[0][0]) + '_' + id2label[res[1][0]])
                                pp_ss.append(id2label[res[1][0]])
                    if i==2:
                        if sent in dd_3:
                            ss.append(str(dd_3[sent][1]) + '_' + dd_3[sent][0])
                            pp_ss.append(dd_3[sent][0])
                        else:
                            if res[0][0] <= 0.5:
                                ss.append("0.0" + '_' + 'None')
                                pp_ss.append('None')
                            else:
                                ss.append(str(res[0][0]) + '_' + id2label[res[1][0]])
                                pp_ss.append(id2label[res[1][0]])
                    if i==0:
                        if res[0][0]<=0.7:
                            ss.append("0.0"+'_'+'None')
                            pp_ss.append('None')
                        else:
                            ss.append(str(res[0][0])+'_'+id2label[res[1][0]])
                            pp_ss.append(id2label[res[1][0]])

                # print(sent,'\t\t',label,'\t\t',ss)
                pp_str='_'.join(pp_ss)

                if pp_str in stand_list:
                    fw.write("True"+'\t\t'+sent + '\t\t' + str(label) + '\n')
                else:
                    fw.write("False"+'\t\t'+sent + '\t\t' + str(label) + '\n')
                    fw_false.write("False"+'\t\t'+sent + '\t\t' + str(label) + '\n')
                    fw_false.write(" ".join(ss) + '\n')
                    fw_false.write('\n')

                fw.write(" ".join(ss) + '\n')
                fw.write('\n')

                if int(label)==3:
                    if pp_str in stand_list:
                        count_dict['3_correct']=count_dict.get('3_correct',0)+1
                    else:
                        count_dict['3_error']=count_dict.get('3_error',0)+1

                if int(label)==1:
                    if pp_str in stand_list:
                        count_dict['1_correct']=count_dict.get('1_correct',0)+1
                    else:
                        count_dict['1_error']=count_dict.get('1_error',0)+1

                if int(label)==2:
                    if pp_str in stand_list:
                        count_dict['2_correct']=count_dict.get('2_correct',0)+1
                    else:
                        count_dict['2_error']=count_dict.get('2_error',0)+1

                if int(label)==5:
                    if pp_str in stand_list:
                        count_dict['5_correct']=count_dict.get('5_correct',0)+1
                    else:
                        count_dict['5_error']=count_dict.get('5_error',0)+1

                print(count_dict)



    if do_predict_other:
        config = tf.ConfigProto(allow_soft_placement=True)
        path='/dockerdata/KeyWordDataSet/rg_train_20190701_1000002.train_ner'
        fw=open('result_analy.txt','w',encoding='utf-8')
        id2label1 = {v: k for k, v in label_vocab1.items()}
        id2label2 = {v: k for k, v in label_vocab2.items()}
        id2label3 = {v: k for k, v in label_vocab3.items()}
        id2label_list=[id2label1,id2label2,id2label3]
        pre_label_list=[]
        with tf.Session(config=config) as sess:
            sess = model.restore(sess,restore_path)
            pdd = PredictDataDeal(vocab=vocab, seq_len=args.seq_len, poss_vocab=poss_vocab)

            for line in open(path,'r',encoding='utf-8').readlines():
                lines=line.replace('\n','').split(":")
                sent,intent=lines[-3],lines[-4]
                if intent in intent_dict:
                    continue
                t1, t1_len, poss = pdd.predict(sent)
                result = model.predict(sess, t1, t1_len, poss)
                ss=[]
                for i, res in enumerate(result):
                    id2label = id2label_list[i]
                    ss.append(str(id2label[res[1][0]])+'_'+str(res[0][0]))
                pre_label_list.append(ss)

                print('%s %s %s'%(intent,sent,ss))
                fw.write(intent+'\t\t'+sent+'\t\t'+str(ss)+'\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        help="mode", type=str, default='')  # vocab_8kw SeqLstm
    parser.add_argument("--config_path",
                        help="config_path", type=str, default=PATH + '/Configs/BaseLstm.config')  # vocab_8kw
    argsConfig = parser.parse_args()

    train(argsConfig)