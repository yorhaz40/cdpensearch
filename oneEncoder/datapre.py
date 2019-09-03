import ast
import glob
import re
from pathlib import Path

import astor
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
import json







def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)


def mergedata():
    #　把code和comment保存成(id,code,comment)的json格式
    commentdata = open("comments_reverse.txt",encoding="utf-8")
    codedata = open("code_reverse.txt",encoding="utf-8")
    commentdata2 = open("commenttfidf.txt",encoding="utf-8")
    output = open("searchdata.json","w",encoding="utf-8")
    data = {}
    for code,comment,comment2 in zip(codedata.readlines(),commentdata.readlines(),commentdata2.readlines()):
        comment = comment.split(" ",1)[1]
        tag = comment2.split(" ",1)[1]
        id = code.split(" ")[0]
        code = code.split(" ",1)[1]
        codetoken = " ".join(tokenize_code(code))
        data["id"] = id
        data["code"] = code
        data["token"] = codetoken
        data["comment"] = comment
        data["tag"] = tag
        output.write(json.dumps(data)+"\n")
    commentdata.close()
    codedata.close()
    output.close()
def dividedata():
    #将数据分成验证集、测试集和训练集
    f = open("data2.json")
    traindoc = open("train.docstring","w",encoding="utf-8")
    traintoken = open("train.function","w",encoding="utf-8")
    testdoc = open("test.docstring","w",encoding="utf-8")
    testtoken = open("test.function","w",encoding="utf-8")
    validdoc = open("valid.docstring","w",encoding="utf-8")
    validtoken=open("valid.function","w",encoding="utf-8")

    data = []
    for d in f.readlines():
        da = json.loads(d)
        data.append(da)
    train,test = train_test_split(list(data),train_size=0.87,shuffle=True,random_state=8081)
    train,valid = train_test_split(train,train_size=0.82,random_state=8081)

    for traindata in train:
        traindoc.write(traindata["comment"].strip()+"\n")
        traintoken.write(traindata["code"]+"\n")
    for validdata in valid:
        validdoc.write(validdata["comment"].strip()+"\n")
        validtoken.write(validdata["code"]+"\n")
    for testdata in test:
        testdoc.write(testdata["comment"].strip()+"\n")
        testtoken.write(testdata["code"]+"\n")
    f.close()
    traindoc.close()
    traintoken.close()
    testdoc.close()
    testtoken.close()
    validdoc.close()
    validtoken.close()

def dividedocstring():
    # 为训练语言模型做准备
    f = open("comments_reverse.txt")
    traindoc = open("ntrain.docstring","w",encoding="utf-8")
    testdoc = open("ntest.docstring","w",encoding="utf-8")
    validdoc = open("nvalid.docstring","w",encoding="utf-8")
    data = []
    for d in f.readlines():
        da = d.split(" ",1)[1]
        data.append(da)
    train,test = train_test_split(list(data),train_size=0.87,shuffle=True,random_state=8081)
    train,valid = train_test_split(train,train_size=0.82,random_state=8081)
    for traindata in train:
        traindoc.write(traindata.strip()+"\n")
    for validdata in valid:
        validdoc.write(validdata.strip()+"\n")
    for testdata in test:
        testdoc.write(testdata.strip()+"\n")
    f.close()
    traindoc.close()
    testdoc.close()
    validdoc.close()

if __name__ =="__main__":
    mergedata()
    # dividedata()
    # dividedocstring()
    print()

    #
    # train,test = train_test_split(list(data),train_size=0.87,shuffle=True,random_state=8081)
    # train,valid = train_test_split(train,train_size=0.82,random_state=8081)
    # train = pd.concat([d for _, d in train]).reset_index(drop=True)
    # valid = pd.concat([d for _, d in valid]).reset_index(drop=True)
    # test = pd.concat([d for _, d in test]).reset_index(drop=True)
