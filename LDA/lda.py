from gensim.models import ldamodel
from gensim import corpora, models
import os
import re
import jieba
base_path=os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
stop_word=[e.replace('\n','') for e in open(base_path+'/stop_word.txt','r',encoding='utf-8').readlines()]


class LDA(object):

    def __init__(self):

        self.patern = 'https://.*/'



    def pre_deal(self,sent):
        sent = str(sent).strip().replace(' ', '')
        sent = re.subn(self.patern, "", sent)[0]
        return sent


    def data_get(self,data_path):

        tokens=[]
        sents=[]
        labels=[]
        for ele in open(data_path,'r',encoding='utf-8').readlines():
            ele=ele.replace('\n','')
            try:
                sent=ele.split('\t\t')[1]
                sent=self.pre_deal(sent)
                sents .append(sent)
                tokens.append([ e for e in jieba.lcut(sent) if e not in stop_word])
            except:
                print(ele)


        # 得到文档-单词矩阵 （直接利用统计词频得到特征）
        dictionary = corpora.Dictionary(tokens)  # 得到单词的ID,统计单词出现的次数以及统计信息
        # print type(dictionary)            # 得到的是gensim.corpora.dictionary.Dictionary的class类型

        texts = [dictionary.doc2bow(text) for text in tokens]  # 将dictionary转化为一个词袋，得到文档-单词矩阵
        texts_tf_idf = models.TfidfModel(texts)[texts]  # 文档的tf-idf形式(训练加转换的模式)
        lda=models.ldamodel.LdaModel.load('./lda.model')
        # lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=3, update_every=0, passes=20,iterations=100)
        texts_lda = lda[texts_tf_idf]
        lda.save('lda.model')
        lda_word=lda.print_topics(num_topics=3, num_words=10)
        for ele in lda_word:
            print(ele)

        for ss,doc in zip(sents,texts_lda):
            print(ss,doc)
            print('\n\n')


if __name__ == '__main__':
    lda=LDA()
    lda.data_get(base_path+'/data/origin_data.txt')