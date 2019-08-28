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



def cleanCommentbyline(comment):
    # try to choose the first sentence of comment.
    # num = comment.split("\t")[0]
    # comment = comment.split("\t")[1]

    # clean the @param and @return words.
    pattern = re.compile("@param.*\*")
    comment = re.sub(pattern, "", comment)
    pattern = re.compile("@return.*\*")
    comment = re.sub(pattern, "", comment)



    #get the first sentence with split from "."
    comment = comment.split(". ")[0] + "."
    comment = comment.replace("/", "")
    comment = comment.replace("*", "")

    comment = " ".join(RegexpTokenizer(r'\w+').tokenize(comment))



    #comment must be longer than 3 words.
    c = comment.split(" ")
    if (len(c) < 4):
        comment = ""
    # chinese character filter.

    # comment.encode('utf-8').decode()

    return comment




def tokenize_code(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)


def mergedata():
    #　把code和comment保存成(id,code,comment)的json格式

    codefile = open("code.data",encoding="utf-8")
    commentfile = open("comment.data",encoding="utf-8")
    apifile = open("api.data",encoding="utf-8")
    seqfile = open("seq.data",encoding="utf-8")



    output = open("searchdata.json","w",encoding="utf-8")
    data = {}
    count = 0
    for code,comment,api,seq in zip(codefile.readlines(),commentfile.readlines(),apifile.readlines(),seqfile.readlines()):

        comment = comment.split("\t",1)[1].strip()
        code = code.split("\t")[1].strip()

        api = api.split("\t")[1].strip()
        seq = seq.split("\t")[1].strip()



        codetoken = " ".join(tokenize_code(code))
        comment = cleanCommentbyline(comment)
        if(seq=="" or api==""):
            continue
        if comment =="":
            continue
        count += 1
        data["id"] = count
        data["code"] = code
        data["token"] = codetoken
        data["seq"] = seq
        data["comment"] = comment
        data["api"] = api
        output.write(json.dumps(data)+"\n")
    codefile.close()
    apifile.close()
    seqfile.close()
    commentfile.close()
    output.close()

def dividedata():
    #将数据分成验证集、测试集和训练集
    f = open("data.json")
    traindoc = open("train.docstring","w",encoding="utf-8")
    trainapi = open("train.api","w",encoding="utf-8")
    trainseq = open("train.seq","w",encoding="utf-8")
    traintoken = open("train.function","w",encoding="utf-8")

    testdoc = open("test.docstring","w",encoding="utf-8")
    testtoken = open("test.function","w",encoding="utf-8")
    testapi = open("test.api","w",encoding="utf-8")
    testseq = open("test.seq","w",encoding="utf-8")

    validdoc = open("valid.docstring","w",encoding="utf-8")
    validtoken=open("valid.function","w",encoding="utf-8")
    validapi = open("valid.api","w",encoding="utf-8")
    validseq = open("valid.seq","w",encoding="utf-8")

    data = []
    for d in f.readlines():
        da = json.loads(d)
        data.append(da)
    train,test = train_test_split(list(data),train_size=0.87,shuffle=True,random_state=8081)
    train,valid = train_test_split(train,train_size=0.82,random_state=8081)

    for traindata in train:
        traindoc.write(traindata["comment"].strip()+"\n")
        traintoken.write(traindata["code"]+"\n")
        trainapi.write(traindata["api"].strip()+"\n")
        trainseq.write(traindata["seq"].strip()+"\n")
    for validdata in valid:
        validdoc.write(validdata["comment"].strip()+"\n")
        validtoken.write(validdata["code"]+"\n")
        validapi.write(validdata["api"]+"\n")
        validseq.write(validdata["seq"]+"\n")
    for testdata in test:
        testdoc.write(testdata["comment"].strip()+"\n")
        testtoken.write(testdata["code"]+"\n")
        testapi.write(testdata["api"].strip()+"\n")
        testseq.write(testdata["seq"].strip()+"\n")
    f.close()
    traindoc.close()
    traintoken.close()
    trainapi.close()
    trainseq.close()
    testdoc.close()
    testtoken.close()
    testapi.close()
    testseq.close()
    validdoc.close()
    validtoken.close()
    validapi.close()
    validseq.close()

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
    # mergedata()
    dividedata()
    # dividedocstring()
    # print()

    #
    # train,test = train_test_split(list(data),train_size=0.87,shuffle=True,random_state=8081)
    # train,valid = train_test_split(train,train_size=0.82,random_state=8081)
    # train = pd.concat([d for _, d in train]).reset_index(drop=True)
    # valid = pd.concat([d for _, d in valid]).reset_index(drop=True)
    # test = pd.concat([d for _, d in test]).reset_index(drop=True)
