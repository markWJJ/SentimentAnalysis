import random
import tensorflow as tf
import numpy as np
import os
import re
import collections
import jieba
import jieba.posseg
from collections import defaultdict
jieba.load_userdict('./user_dict.txt')
STOP_WORD=[e for e in open('./stop_word.txt','r',encoding='utf-8').readlines()]

class DataSetTfrecord(object):
    def __init__(self, corpus_path, vocab, seq_len,poss_vocab=None,intent_dict=None,out_path=None,vocab_char=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.intent_dict=intent_dict
        self.corpus_path=corpus_path
        self.Poss_dict=poss_vocab
        self.id2labels=[]
        self.id2vocab={v:k for k,v in vocab.stoi.items()}
        self.trans2tfRecord(output_path=out_path)



    def getSegPossIfo(self,sent):
        sentence_seged = jieba.posseg.cut(sent)
        char_info=[]
        poss_info=[]
        char_info_origin=[]
        poss_info_origin=[]
        for x in sentence_seged:
            word,poss=x.word, x.flag
            if len(word)==1:
                poss_info.append(self.Poss_dict.get(poss,self.Poss_dict['None']))
                char_info_origin.append("S")
                poss_info_origin.append(poss)
            else:
                poss_info.extend([self.Poss_dict.get(poss,self.Poss_dict['None'])]*len(word))
                cs=["I"]*len(word)
                cs[0]="B"
                cs[-1]="E"
                char_info_origin.extend(cs)
                poss_info_origin.extend([poss]*len(word))
        return poss_info,poss_info_origin

    def get_key_info(self,key_dict,sents,intent):
        if intent in key_dict:
            keys=key_dict[intent]
            key_emb=[]
            for word in sents:
                if word in keys:
                    key_emb.append(5)
                else:
                    key_emb.append(1)
            key_emb=[1]+key_emb+[1]
            key_emb.extend([0]*self.seq_len)
        else:
            key_emb=[5]*self.seq_len
        return key_emb[:self.seq_len]



    def trans2tfRecord(self,output_path=None):
        '''
        构建tfrecord
        :param output_path:
        :return:
        '''
        filename = output_path
        writer = tf.python_io.TFRecordWriter(filename)
        writer_index=open(output_path+".index",'w',encoding='utf-8')
        index=0
        id2vocab = {v: k for k, v in self.vocab.stoi.items()}
        datas=[]
        with open(self.corpus_path,'r',encoding='utf-8') as fr:
            for line in fr:
                datas.append(line)

        random.shuffle(datas)

        for line in datas:
            lines=line.replace('\n','').split('\t\t')
            intent,t1=lines[0],lines[1]
            poss_info,poss_info_origin=self.getSegPossIfo(t1)
            intent=self.intent_dict.get(intent,self.intent_dict['None'])

            t1=[e for e in t1]
            t1 = self.vocab.to_seq(t1)
            t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
            t1_len = min(len(t1), self.seq_len)
            padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
            t1.extend(padding_t1)
            t1= t1[:self.seq_len]

            poss = poss_info
            poss = [self.Poss_dict["None"]] + poss + [self.Poss_dict["None"]]
            padding_poss = [self.Poss_dict["None"] for _ in range(self.seq_len - len(poss))]
            poss.extend(padding_poss)
            poss = poss[:self.seq_len]

            features = collections.OrderedDict()
            features["sent_word"] = self._int64_feature(t1)
            features["sent_len"] = self._int64_feature([t1_len])
            features["poss"] = self._int64_feature(poss)
            features["intent"] = self._int64_feature([intent])

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            index+=1
            if index%5000==0:
                print(index)
                print([id2vocab.get(e,'NONE') for e in t1])
                print(t1,'\t\t',t1_len,'\t\t',poss)
                print([self.id2vocab[e] for e in t1])
        writer.close()
        writer_index.write(str(index))
        writer_index.close()


    def _int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

    def _int64_feature1(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def predict(self,t1):
        t1 = self.vocab.to_seq(t1)
        t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
        t1_len = min(len(t1), self.seq_len)
        padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
        t1.extend(padding_t1)
        t1 = t1[:self.seq_len]

        return np.expand_dims(np.array(t1),0),np.array(t1_len)


