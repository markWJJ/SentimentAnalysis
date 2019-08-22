import numpy as np
import jieba.posseg
jieba.load_userdict('./user_dict.txt')
import re
class PredictDataDeal(object):
    def __init__(self, vocab, seq_len, label_vocab=None,ner_vocab=None,seg_vocab=None,poss_vocab=None,vocab_char=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.label_vocab=label_vocab
        self.ner_vocab=ner_vocab
        self.vocab_char=vocab_char
        self.seg_vocab=seg_vocab
        self.poss_vocab=poss_vocab
        self.entity_list=["酷狗音乐","华为云","微信读书","银河奇异果","爱奇艺","酷我音乐"
                          ,"酷狗","土豆","陌陌","qq音乐","乐视网","滴滴打车","qq超级会员","好莱坞","腾讯视频","qq会员"
                          ,"超级会员" ,"QQ会员","QQ黄钻","拼多多","网易云音乐会员","芒果TV会员",".{1,2}会员","愛奇异"]
        self.entity_list=[]
        self.entity_list_re="|".join(self.entity_list)

        self.synonym={"关闭":["关比","关毕"],
                      }

        self.synonym_re=[["|".join(v),k] for k,v in self.synonym.items()]

    def getSegPossIfo(self, sent):
        sentence_seged = jieba.posseg.cut(sent)
        char_info = []
        poss_info = []
        char_info_origin = []
        poss_info_origin = []
        for x in sentence_seged:
            word, poss = x.word, x.flag
            if len(word) == 1:
                poss_info.append(self.poss_vocab.get(poss, self.poss_vocab['None']))
                poss_info_origin.append(poss)
            else:

                poss_info.extend([self.poss_vocab.get(poss, self.poss_vocab['None'])] * len(word))
                poss_info_origin.extend([poss] * len(word))
        return  poss_info,  poss_info_origin
    
    def predict(self,t1):
        poss_info,  poss_info_origin = self.getSegPossIfo(t1)
        for ele in self.synonym_re:
            t1=re.subn(ele[0],ele[1],t1)[0]
        t1_remove_entity = re.subn(self.entity_list_re, "", t1)[0]
        print(t1_remove_entity)

        # t1=[e for e in t1]
        origin_t1=t1
        t1=jieba.lcut(t1)
        t1 = self.vocab.to_seq(t1)
        t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
        t1_len = min(len(t1), self.seq_len)
        padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
        t1.extend(padding_t1)
        t1 = t1[:self.seq_len]

        t1_char = [e for e in origin_t1]
        t1_char = self.vocab.to_seq(t1_char)
        t1_char = [self.vocab.sos_index] + t1_char + [self.vocab.eos_index]
        t1_char_len = min(len(t1_char), self.seq_len)
        padding_t1_char = [self.vocab.pad_index for _ in range(self.seq_len - len(t1_char))]
        t1_char.extend(padding_t1_char)
        t1_char = t1_char[:self.seq_len]

        # t1_remove_entity = [e for e in t1_remove_entity]
        origin_t1_remove_entity=t1_remove_entity
        t1_remove_entity = jieba.lcut(t1_remove_entity)
        t1_remove_entity = self.vocab.to_seq(t1_remove_entity)
        t1_remove_entity = [self.vocab.sos_index] + t1_remove_entity + [self.vocab.eos_index]
        t1_remove_entity_len = min(len(t1_remove_entity), self.seq_len)
        padding_t1_en = [self.vocab.pad_index for _ in range(self.seq_len - len(t1_remove_entity))]
        t1_remove_entity.extend(padding_t1_en)
        t1_remove_entity = t1_remove_entity[:self.seq_len]

        t1_remove_entity_char = [e for e in origin_t1_remove_entity]
        t1_remove_entity_char = self.vocab.to_seq(t1_remove_entity_char)
        t1_remove_entity_char = [self.vocab.sos_index] + t1_remove_entity_char + [self.vocab.eos_index]
        t1_remove_entity_char_len = min(len(t1_remove_entity_char), self.seq_len)
        padding_t1_en_char = [self.vocab.pad_index for _ in range(self.seq_len - len(t1_remove_entity_char))]
        t1_remove_entity_char.extend(padding_t1_en_char)
        t1_remove_entity_char = t1_remove_entity_char[:self.seq_len]


        poss = poss_info
        poss = [self.poss_vocab["None"]] + poss + [self.poss_vocab["None"]]
        padding_poss = [self.poss_vocab["None"] for _ in range(self.seq_len - len(poss))]
        poss.extend(padding_poss)
        poss = poss[:self.seq_len]


        return np.expand_dims(np.array(t1),0),np.expand_dims(np.array(t1_len),0),np.expand_dims(np.array(poss),0),np.expand_dims(np.array(t1_remove_entity),0),\
               np.expand_dims(np.array(t1_remove_entity_len),0),np.expand_dims(np.array(t1_char),0),np.expand_dims(np.array(t1_char_len), 0),np.expand_dims(np.array(t1_remove_entity_char),0),\
               np.expand_dims(np.array(t1_remove_entity_char_len),0)



