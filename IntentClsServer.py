#!/usr/bin/python3
# coding: utf-8
from tornado.escape import json_decode, json_encode, utf8
import tornado.ioloop
import tornado.web
import json
import logging
import argparse
import os,sys
from IntentCls_API import IntentCls
from urllib import parse

import gc
PATH=os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0,PATH+'/intent_re/intent_de')
PATH='.'
import logging
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logfile='ematic_match.txt'
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

HOST='9.73.128.246'
DEFAULT_PORT=18088
# TREE=Tree(PATH+'/data/意图识别.txt')
parser = argparse.ArgumentParser()

# 设置启动端口：
default_port = DEFAULT_PORT
parser.add_argument('-port', '--port', type=int, default=default_port, help='服务端口，默认: {}'.format(default_port))
args = parser.parse_args()
PORT = args.port

restore_path = "/dockerdata/junjiangwu/Intent3cls/out_model/leam/LEAM_Intent_2kw.ckpt-31"
IC=IntentCls(restore_path)

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil,os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

# @profile
def semantic(sent):
    re_dict=IC.predict_api(sent)
    return re_dict



class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # # self.write("请使用post方法")
        # sent = self.get_argument('sent' )
        # _logger.info(sent)
        # sent=parse.unquote(sent)
        # # print(type(annoy_key))
        # # print(annoy_key)
        #
        # ret = semantic(sent)
        # # result_str=json.dump(ret[0])
        # # print(ret[0])
        # result_str = json.dumps(ret)
        # self.write(result_str)
        # sys.stdout.flush()
        pass

    def post(self):
        # try:
            # if 'application/json' in content_type.lower():
        body_data = self.request.body
        if isinstance(body_data, bytes):
            body_data = self.request.body.decode('utf8')

        args_data = json.loads(body_data)
        data = args_data.get('sent', [])
        _logger.info(data)
        ret = semantic(data)
        ret={'result':ret}
        result_str = json.dumps(ret, ensure_ascii=False)
        print(result_str)
        self.write(result_str)
        # except Exception as e:
        #     self.write('{}')
        #     self.write("\n")
        sys.stdout.flush()

def make_app():
    return tornado.web.Application([
        (r"/nlp/IntentCls", MainHandler),
    ])

def main():
    app = make_app()
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()

# svr=SimpleXMLRPCServer((HOST, PORT), allow_none=True)
# svr.register_function(intent)
# svr.serve_forever()

# if __name__ == '__main__':
#
#     ss=[e.replace('\n','') for e in open('./FAQ_1.txt','r').readlines()]
#     sss=ss*20
#     intent(sent_list=sss)