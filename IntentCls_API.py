
import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from Predict import PredictDataDeal
import xlrd
from dataset import WordVocab
import json
import jieba
from collections import defaultdict
jieba.load_userdict('./user_dict.txt')
KEY_WORD_DICT=defaultdict(list)
for e in open('./key_word.txt','r',encoding='utf-8').readlines():
    key,v=e.replace('\n','').split(':')
    KEY_WORD_DICT[key]=[ee for ee in v.split(" ")]

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
PATH=os.path.split(os.path.realpath(__file__))[0]

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


class IntentCls(object):

    def __init__(self,model_name='/'):
        self.sess_dict={}
        self.queryObj_dict = {}
        config = tf.ConfigProto(allow_soft_placement=True)

        model_name_meta = '%s.meta' % model_name
        saver = tf.train.import_meta_graph(model_name_meta)  # 加载图结构
        gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
        tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
        for ele in tensor_name_list:
            if str(ele).__contains__('out_softmax'):
                print(ele)
        args_dict = json.load(open('./Configs/BaseLstm.config', 'r'))
        self.args = dict_to_object(dict(args_dict))
        self.label_vocab1 = pkl.load(open("./label_vocab1.p", 'rb'))
        self.label_vocab2 = pkl.load(open("./label_vocab2.p", 'rb'))
        self.label_vocab3 = pkl.load(open("./label_vocab3.p", 'rb'))
        self.vocab = WordVocab.load_vocab(PATH+self.args.vocab_path)
        self.poss_vocab = pkl.load(open("./poss_vocab.p", 'rb'))

        with tf.device('/device:GPU:%s' % 0):
            self.sent_token = gragh.get_tensor_by_name('sent1_token:0')
            self.sent_char = gragh.get_tensor_by_name('sent1_char:0')

            self.sent_word_re =gragh.get_tensor_by_name('sent_word_re:0')
            self.sent_word_re_char = gragh.get_tensor_by_name('sent_word_re_char:0')

            self.sent_len = gragh.get_tensor_by_name('sent1_len:0')
            self.sent_len_char =gragh.get_tensor_by_name('sent_len_char:0')

            self.sent_len_re = gragh.get_tensor_by_name('sent1_len_re:0')
            self.sent_len_re_char = gragh.get_tensor_by_name('sent1_len_re_char:0')

            self.sent_token_neg = gragh.get_tensor_by_name('sent1_token_neg:0')
            self.sent_len_neg = gragh.get_tensor_by_name('sent1_len_neg:0')
            self.sent_char_neg = gragh.get_tensor_by_name('sent_char_neg:0')
            self.sent_char_len_neg = gragh.get_tensor_by_name('sent_char_len_neg:0')

            self.key_emb=gragh.get_tensor_by_name('key_emb:0')

            self.dropout = gragh.get_tensor_by_name('dropout:0')

            name=model_name.split('/')[-1].split('_')[0].replace("BaseLSTM","BaseLstm")
            try:
                self.soft_out_1 = gragh.get_tensor_by_name('%s_enc_0/_0/softmax/Softmax:0'%name)
                self.soft_out_2 = gragh.get_tensor_by_name('%s_enc_1/_1/softmax/Softmax:0'%name)
                self.soft_out_3 = gragh.get_tensor_by_name('%s_enc_2/_2/softmax/Softmax:0'%name)
            except:
                self.soft_out_1 = gragh.get_tensor_by_name('%s_enc_0/_0/out_softmax/softmax/Softmax:0' % name)
                self.soft_out_2 = gragh.get_tensor_by_name('%s_enc_1/_1/out_softmax/softmax/Softmax:0' % name)
                self.soft_out_3 = gragh.get_tensor_by_name('%s_enc_2/_2/out_softmax/softmax/Softmax:0' % name)

            try:
                self.smentic_out_1 = gragh.get_tensor_by_name('%s_enc_0/_0/semantic_out/concat:0'%name)
                self.smentic_out_2 = gragh.get_tensor_by_name('%s_enc_1/_1/semantic_out/concat:0' % name)
                self.smentic_out_3 = gragh.get_tensor_by_name('%s_enc_2/_2/semantic_out/concat:0' % name)
            except:
                pass

            self.sess = tf.Session(config=config)
            saver.restore(self.sess, '%s' % model_name)

            self.pdd = PredictDataDeal(vocab=self.vocab, seq_len=self.args.seq_len, poss_vocab=self.poss_vocab)

    def get_key_info(self, key_dict, sents, intent):
        if intent in key_dict:
            keys = key_dict[intent]
            key_emb = []
            for word in sents:
                if word in keys:
                    key_emb.append(5)
                else:
                    key_emb.append(1)
            key_emb = [1] + key_emb + [1]
            key_emb.extend([0] * self.args.seq_len)
        else:
            key_emb = [5] * self.args.seq_len
        return key_emb[:self.args.seq_len]

    def predict(self,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len,key_emb):
        feed_dict = {
            self.sent_token: t1,
            self.sent_len: t1_len,
            self.sent_word_re: t1_re,
            self.sent_len_re: t1_re_len,
            self.sent_char: t1_char,
            self.sent_len_char: t1_char_len,
            self.sent_word_re_char: t1_rm_char,
            self.sent_len_re_char: t1_rm_char_len,
            self.key_emb:key_emb,
            self.dropout: 0.0
        }
        # print(sent1_word_)

        pre1, pre2, pre3 = self.sess.run([self.soft_out_1, self.soft_out_2, self.soft_out_3],
                                    feed_dict=feed_dict)
        res = []
        for ele in [pre1, pre2, pre3]:
            score = np.max(ele, 1)
            pre_leb = np.argmax(ele, 1)
            res.append([score, pre_leb])
        return res


    def predict_sent(self):
        id2label1 = {v: k for k, v in self.label_vocab1.items()}
        id2label2 = {v: k for k, v in  self.label_vocab2.items()}
        id2label3 = {v: k for k, v in  self.label_vocab3.items()}
        id2label_list=[id2label1,id2label2,id2label3]
        pdd = PredictDataDeal(vocab=self.vocab, seq_len=self.args.seq_len, poss_vocab=self.poss_vocab)
        while True:
            ss = []
            sent = input("输入:")
            print(jieba.lcut(sent))
            key_emb=self.get_key_info(KEY_WORD_DICT,jieba.lcut(sent),'4000142')
            key_emb=np.expand_dims(np.array(key_emb),0)
            t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len = pdd.predict(sent)
            result = self.predict(t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len,key_emb)

            for i, res in enumerate(result):
                id2label = id2label_list[i]
                # print(res[0][0])
                # print(id2label[res[1][0]])
                ss.append("_".join([str(res[0][0]), str(id2label[res[1][0]])]))
            print(ss)

    def predict_api(self,sent):
        id2label1 = {v: k for k, v in self.label_vocab1.items()}
        id2label2 = {v: k for k, v in self.label_vocab2.items()}
        id2label3 = {v: k for k, v in self.label_vocab3.items()}
        id2label_list = [id2label1, id2label2, id2label3]
        key_emb = self.get_key_info(KEY_WORD_DICT, jieba.lcut(sent), '4000142')
        key_emb = np.expand_dims(np.array(key_emb), 0)
        t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len = self.pdd.predict(sent)
        result = self.predict(t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len,
                              key_emb)
        result_finall=[]
        dd_dict={}
        for i, res in enumerate(result):
            id2label = id2label_list[i]
            dd_dict['label_%s'%i]=[id2label[res[1][0]],str(res[0][0])]
            # result_finall.append([id2label[res[1][0]]res[0][0]])
        return dd_dict

    def predict_txt(self):
        id2label1 = {v: k for k, v in self.label_vocab1.items()}
        id2label2 = {v: k for k, v in  self.label_vocab2.items()}
        id2label3 = {v: k for k, v in  self.label_vocab3.items()}
        pdd = PredictDataDeal(vocab=self.vocab, seq_len=self.args.seq_len,poss_vocab=self.poss_vocab)
        all_num,correct_num=0,0
        for line in open('./test_with_c1_03.txt','r',encoding='utf-8').readlines():
            lines=line.replace('\n','').split('\t\t')
            sent,label,label_word=lines[3::]
            key_emb=self.get_key_info(KEY_WORD_DICT,jieba.lcut(sent),'4000142')
            key_emb=np.expand_dims(np.array(key_emb),0)
            if label_word=='异常':
                all_num+=1
                t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len = pdd.predict(sent)
                result = self.predict(t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char,
                                      t1_rm_char_len, key_emb)

                print(id2label3[result[2][1][0]])
                if id2label3[result[2][1][0]]=='失败':
                    correct_num+=1
                    print(sent, label_word,"失败")
                else:
                    print(sent, label_word,result)
        print(correct_num,all_num)


    def predict_cls_semantic(self):
        id2label1 = {v: k for k, v in self.label_vocab1.items()}
        id2label2 = {v: k for k, v in self.label_vocab2.items()}
        id2label3 = {v: k for k, v in self.label_vocab3.items()}
        id2label_list = [id2label1, id2label2, id2label3]
        pdd = PredictDataDeal(vocab=self.vocab, seq_len=self.args.seq_len, poss_vocab=self.poss_vocab)
        while True:
            ss = []
            sent1 = input("输入sent1:")
            sent2 = input("输入sent2:")

            print(jieba.lcut(sent1))
            key_emb = self.get_key_info(KEY_WORD_DICT, jieba.lcut(sent1), '4000142')
            key_emb = np.expand_dims(np.array(key_emb), 0)
            t1_sent1, t1_len_sent1, poss_sent1, t1_re_sent1, t1_re_len_sent1, t1_char_sent1, t1_char_len_sent1, t1_rm_char_sent1, t1_rm_char_len_sent1 \
                = pdd.predict(sent1)
            t1_sent2, t1_len_sent2, poss_sent2, t1_re_sent2, t1_re_len_sent2, t1_char_sent2, t1_char_len_sent2, t1_rm_char_sent2, t1_rm_char_len_sent2 \
                = pdd.predict(sent2)

            feed_dict = {
                self.sent_token_neg: t1_sent2,
                self.sent_len_neg: t1_len_sent2,
                self.sent_char_neg: t1_char_sent2,
                self.sent_char_len_neg: t1_char_len_sent2,
                self.sent_token: t1_sent1,
                self.sent_char: t1_char_sent1,
                self.sent_len_char: t1_char_len_sent1,
                self.sent_len: t1_len_sent1,
                self.dropout: 0.0
            }
            # print(sent1_word_)

            s1,s2,s3,pre1, pre2, pre3 = self.sess.run([self.smentic_out_1,self.smentic_out_2,self.smentic_out_3,self.soft_out_1, self.soft_out_2, self.soft_out_3],
                                             feed_dict=feed_dict)

            print(s1,s2,s3)
            res = []
            for ele in [pre1, pre2, pre3]:
                score = np.max(ele, 1)
                pre_leb = np.argmax(ele, 1)
                res.append([score, pre_leb])

            for i, res in enumerate(res):
                id2label = id2label_list[i]
                # print(res[0][0])
                # print(id2label[res[1][0]])
                ss.append("_".join([str(res[0][0]), str(id2label[res[1][0]])]))
            print(ss)

    def predict_xlsx(self):
        dd_2 = {}
        for ele in open('./test_with_c1_02.txt', 'r', encoding='utf-8').readlines():
            eles = ele.replace('\n', '').split('\t\t')
            sent = eles[3]
            score = eles[2]
            label = eles[1]
            if label.__contains__('inter') or label.__contains__('自动续费'):
                label = "None"
            dd_2[sent] = [label, score]

        dd_3 = {}
        for ele in open('./test_03.txt', 'r', encoding='utf-8').readlines():
            eles = ele.replace('\n', '').split('\t\t')
            sent = eles[3]
            score = eles[2]
            label = eles[1]
            if label.__contains__('inter') or label.__contains__('自动续费'):
                label = "None"
            dd_3[sent] = [label, score]

        dd_2={}
        dd_3={}
        # config = tf.ConfigProto(allow_soft_placement=True)
        path = './cp_4000142_0726.xlsx'
        # path = './红包退回规则.xlsx'

        id2label1 = {v: k for k, v in self.label_vocab1.items()}
        id2label2 = {v: k for k, v in self.label_vocab2.items()}
        id2label3 = {v: k for k, v in self.label_vocab3.items()}
        id2label_list = [id2label1, id2label2, id2label3]
        true_label_list = []
        pre_label_list = []
        fw = open("cp_4000142_0726_自动续费.txt", 'w', encoding='utf-8')
        fw_false = open("cp_4000142_0726_自动续费_false.txt", 'w', encoding='utf-8')

        count_dict = {}
        # with tf.Session(config=config) as sess:
        # sess = model.restore(sess, restore_path)
        pdd = PredictDataDeal(vocab=self.vocab, seq_len=self.args.seq_len, poss_vocab=self.poss_vocab)

        work_sheet = xlrd.open_workbook(path)
        sheet = work_sheet.sheet_by_index(1)
        stand_list = ["自动续费_撤销_None", "自动续费_None_None", "自动续费_退款_None", "自动续费_开通_None", "自动续费_查询_None","自动续费_退回规则_None"]
        # stand_list = ["红包_退款_None", "红包_退回规则_None"," 红包_退款_未入账"]

        stand_list=[str(e).lower() for e in stand_list]
        for i in range(1, sheet.nrows):
            try:
                sent = sheet.cell_value(i, 2)
                label = sheet.cell_value(i, 3)
            except:
                sent = sheet.cell_value(i, 0)
                label = sheet.cell_value(i, 1)
            # entity = entity_detect(sent)
            flag = True
            t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len = pdd.predict(sent)
            # result=model.predict(sess,t1,t1_len,poss,t1_re,t1_re_len,t1_char,t1_char_len,t1_rm_char,t1_rm_char_len)
            key_emb=self.get_key_info(KEY_WORD_DICT,jieba.lcut(sent),'4000142')
            key_emb=np.expand_dims(np.array(key_emb),0)
            result = self.predict(t1, t1_len, poss, t1_re, t1_re_len, t1_char, t1_char_len, t1_rm_char, t1_rm_char_len,key_emb)
            ss = []
            pp_ss = []
            for j, res in enumerate(result):
                id2label = id2label_list[j]
                # print(res[0][0])
                # print(id2label[res[1][0]])
                if j == 1:
                    if sent in dd_2:
                        ss.append(str(dd_2[sent][1]) + '_' + dd_2[sent][0])
                        pp_ss.append(dd_2[sent][0])
                    else:
                        if res[0][0] <= 0.8:
                            ss.append("0.0" + '_' + 'None')
                            pp_ss.append('None')
                        else:
                            ss.append(str(res[0][0]) + '_' + id2label[res[1][0]])
                            pp_ss.append(id2label[res[1][0]])
                if j == 2:
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
                if j==0:
                    if res[0][0] <= 0.5:
                        ss.append("0.0" + '_' + 'None')
                        pp_ss.append('None')
                    else:
                        ss.append(str(res[0][0]) + '_' + id2label[res[1][0]])
                        pp_ss.append(id2label[res[1][0]])

            pp_str = '_'.join(pp_ss).lower()
            print(sent,'\t\t',pp_str)
            if pp_str in stand_list:
                fw.write("True" + '\t\t' + sent + '\t\t' + str(label) + '\n')
                if int(label) == 3:
                    fw_false.write("True" + '\t\t' + sent + '\t\t' + str(label) + '\n')
                    fw_false.write(" ".join(ss) + '\n')
                    fw_false.write('\n')
            else:
                if int(label) in [1,2,5]:
                    fw.write("False" + '\t\t' + sent + '\t\t' + str(label) + '\n')
                    fw_false.write("False" + '\t\t' + sent + '\t\t' + str(label) + '\n')
                    fw_false.write(" ".join(ss) + '\n')
                    fw_false.write('\n')

            fw.write(" ".join(ss) + '\n')
            fw.write('\n')

            if int(label) == 3:
                if pp_str in stand_list:
                    count_dict['3_correct'] = count_dict.get('3_correct', 0) + 1
                else:
                    count_dict['3_error'] = count_dict.get('3_error', 0) + 1

            if int(label) == 1:
                if pp_str in stand_list:
                    count_dict['1_correct'] = count_dict.get('1_correct', 0) + 1
                else:
                    count_dict['1_error'] = count_dict.get('1_error', 0) + 1

            if int(label) == 2:
                if pp_str in stand_list:
                    count_dict['2_correct'] = count_dict.get('2_correct', 0) + 1
                else:
                    count_dict['2_error'] = count_dict.get('2_error', 0) + 1

            if int(label) == 5:
                if pp_str in stand_list:
                    count_dict['5_correct'] = count_dict.get('5_correct', 0) + 1
                else:
                    count_dict['5_error'] = count_dict.get('5_error', 0) + 1

        print(count_dict)


if __name__ == '__main__':
    # restore_path="/dockerdata/junjiangwu/Intent3cls/out_model/lstm/BaseLSTM_Intent_2kw.ckpt-9"
    # restore_path="/dockerdata/junjiangwu/Intent3cls/out_model/transformer/BaseLstmStruct_Intent_2kw.ckpt-18"
    # restore_path="/dockerdata/junjiangwu/Intent3cls/out_model/cnn/cnn_Intent_2kw.ckpt-31"
    # restore_path="/dockerdata/junjiangwu/Intent3cls/out_model/transformer_cnn/TransformerCNN_Intent_2kw.ckpt-29"
    restore_path="/dockerdata/junjiangwu/Intent3cls/out_model/leam/LEAM_Intent_2kw.ckpt-31"

    ic=IntentCls(restore_path)
    ic.predict_sent()
    # ic.predict_txt()
    # ic.predict_xlsx()
    # ic.predict_cls_semantic()
