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

    def getSegPossIfo(self, sent):
        sentence_seged = jieba.posseg.cut(sent)
        poss_info = []
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

        t1 = [e for e in t1]
        t1 = self.vocab.to_seq(t1)
        t1 = [self.vocab.sos_index] + t1 + [self.vocab.eos_index]
        t1_len = min(len(t1), self.seq_len)
        padding_t1 = [self.vocab.pad_index for _ in range(self.seq_len - len(t1))]
        t1.extend(padding_t1)
        t1 = t1[:self.seq_len]

        poss = poss_info
        poss = [self.poss_vocab["None"]] + poss + [self.poss_vocab["None"]]
        padding_poss = [self.poss_vocab["None"] for _ in range(self.seq_len - len(poss))]
        poss.extend(padding_poss)
        poss = poss[:self.seq_len]


        return np.expand_dims(np.array(t1),0),np.expand_dims(np.array(t1_len),0),np.expand_dims(np.array(poss),0)



