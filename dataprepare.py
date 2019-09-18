import os



import re
import json

from nltk.tokenize import RegexpTokenizer

def split_data_normal(ori_data):
    ## 首先判断是否有大写
    word_list = []
    if not (re.search(r'[A-Z]', ori_data)):
        word_list.append(ori_data)
    else:
        total_length = len(ori_data)
        temp_flag = 0
        continue_flag = 0
        small_flag = 0
        for i in range(total_length):
            if ori_data[i].isupper():
                if (temp_flag != i and (i - temp_flag) > 1):
                    if (continue_flag != 1):
                        word_list.append(ori_data[temp_flag:i])
                        temp_flag = i
                elif (temp_flag != i and (i - temp_flag) == 1):
                    if (temp_flag == small_flag):
                        if (small_flag == 0 and ori_data[0].isupper()):
                            continue_flag = 1
                        else:
                            word_list.append(ori_data[temp_flag:i])
                            temp_flag = i
                    else:
                        continue_flag = 1

            else:
                if (continue_flag == 1):
                    continue_flag = 0
                    word_list.append(ori_data[temp_flag:i])
                    temp_flag = i
                    small_flag = i
        word_list.append(ori_data[temp_flag:])
    return (word_list)

def split_data_myself(ori_data):
    pass
def deal_comments(string):
    string = string.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    string = string.replace("EOF","").replace("*", " ").replace("@", " @ ").replace(",", " , ").replace(".", " . ").replace(":"," : ").replace("'", " ' ")
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", "  ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", "  ", string)
    string = re.sub(r"\"", "  ", string)  # "a"       --> " a "
    string = re.sub(r"\'", "  ", string)  # they'     --> they '
    string = re.sub(r"\’", "  ", string)  # they’     --> they ’
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\s{2,}", " ", string)  # Akara is    handsome --> Akara is handsome
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", " “ ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", " ” ", string)
    string = re.sub(r"\"", " “ ", string)  # "a"       --> " a "
    string = re.sub(r"\'", " ' ", string)  # they'     --> they '
    string = re.sub(r"\’", " ' ", string)  # they’     --> they '
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\-", " ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", " ( ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", " ) ", string)  # ( they)   --> ( they )
    string = re.sub(r"\!", " ! ", string)  # they!     --> they !
    string = re.sub(r"\]", " ] ", string)  # they]     --> they ]
    string = re.sub(r"\[", " [ ", string)  # they[     --> they [
  #  string = re.sub(r"\?", " ? ", string)  # they?     --> they ?
    string = re.sub(r"\>", " > ", string)  # they>     --> they >
    string = re.sub(r"\<", " < ", string)  # they<     --> they <
    string = re.sub(r"\=", " = ", string)  # easier=   --> easier =
    string = re.sub(r"\;", " ; ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"}", " } ", string)
    string = re.sub(r"{", " { ", string)
    string = re.sub(r"\:", " : ", string)  # easier:   --> easier :
    string = re.sub(r"\"", " \" ", string)  # easier"   --> easier "
    string = re.sub(r"\$", " $ ", string)  # $380      --> $ 380
    string = re.sub(r"\_", " _ ", string)  # _100     --> _ 100
    string = re.sub(r"\s{2,}", " ", string)  # Akara is    handsome --> Akara is handsome
    # string = re.sub(r"\s{2,}", " ", string)
    string=split_data_normal(string)
    string=(" ").join(string)
    string = string.strip().lower()  # lowercase
    string=(" ").join(string.split())
    return string



def deal_code(string):
    string = re.sub(r"//.*\n","",string)
    string=split_data_normal(string)
    string=(" ").join(string)
    string = string.strip().lower()  # lowercase
    string=(" ").join(string.split())
    return string



def cleandata(path,output):
    ##清理代码中的注释文字，并清理注释中的一些字符，将所有处理过后的文字存储在一个json中
    flist = os.listdir(path)#因为数据存储在分开的许多小文件中
    id = 0
    outfile = open(output,"w",encoding="utf-8")
    for f in flist:
        file = open(path+"/"+f,encoding="utf-8")
        for line in file:
            data = json.loads(line)


            comment = data["comment"]
            code = data["code"]
            ifmatch = re.match(r".*[\u4e00-\u9fa5].*", comment)
            if ifmatch!=None:
                continue
            # else:
            #     print(f+" : "+comment)
            if len(comment.split(" "))<5:#小于5的注释直接跳过
                continue
            id += 1
            dcomment = " ".join(RegexpTokenizer(r'\w+').tokenize(comment))
            # dcomment = deal_comments(comment)
            dcode = code.replace(comment.strip(),"")
            # dcode = deal_code(dcode)
            ddata = {}
            ddata["id"] = id
            ddata["comment"] = dcomment
            ddata["code"] = dcode
            outfile.write(json.dumps(ddata)+"\n")
        file.close()
    outfile.close()




def cleanfinal(path):
    file = open(path)
    outfile = open("data_final.json","w")
    for line in file:
        data = json.loads(line)

        comment = data["comment"]

        ifmatch = re.match(r".*[\u4e00-\u9fa5].*", comment)
        if ifmatch != None:
            print(comment)
            continue
        outfile.write(json.dumps(data)+"\n")






from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer


from nltk.corpus import stopwords




def remove_key_words(open_key):
    key_words_list=[]
    for i in open_key:
         key_words_list.append(i.replace("\n",""))
    for i  in  key_words_list:
        print (i)
    open_code_new1=open("code_new.txt","r")
    open_code_deal=open("code_new_deal.txt","a+")
    for i in open_code_new1:
        print (i.replace("\n","").split(" ")[0:])



def deal_tfidf_pro(path):
    open_key = open("key_words.txt", "r", encoding="utf-8")
    output = open("tfidfdata","w",encoding="utf-8")
    total_list=[]
    number_list=[]
    key_words_list = []
    for i in open_key:
        key_words_list.append(i.replace("\n", ""))
    file = open(path)
    count = 0
    for line in file:

        data = json.loads(line)
        i = data["code"]
        id = data["id"]
        after_deal=(i.replace("\n","").replace("<","").replace(">",""))
        after_deal=(after_deal.split(" "))
        real_tem_list=[]
        for jj in after_deal:
            if jj in key_words_list:
                pass
            else:
                real_tem_list.append(jj)
        after_deal=(" ").join(real_tem_list)
        number_list.append(id)
        total_list.append(after_deal)
        count += 1
        if(count%10000 == 0):
            print(count)
    X = vectorizer.fit_transform(total_list)
    #   print (X.toarray()[1])
    tfidf = transformer.fit_transform(X)
    full_word_list = vectorizer.get_feature_names()
    total_len = tfidf.shape[0]
    for i in range(total_len):
        temp_array = tfidf[i].toarray()[0]
        temp_array = list(temp_array)
        temp_dict = {}
        number_big = {}
        for ii, jj in enumerate(temp_array):
            if (jj == 0):
                pass
            else:
                number_big[ii] = jj
        number_big = sorted(number_big.items(), key=lambda x: x[1], reverse=True)
        for qq in number_big[:10]:
            temp_words = full_word_list[qq[0]]
            if temp_words not in cachedStopwords and (str(temp_words).isdigit() == False):
                temp_dict[temp_words] = (total_list[i].index(temp_words))
        temp_dict = sorted(temp_dict.items(), key=lambda x: x[1], reverse=False)
        temp_string = []
        for qq in temp_dict:
            temp_string.append(qq[0])
        final_string = (" ").join(temp_string)
        output.writelines(str(number_list[i]) + " " + final_string + "\n")
        if(i%10000 == 0):
            print(i)
    output.close()
    file.close()


from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer

if __name__ == "__main__":
    ##去除代码中的多余字符，去除comment中的多余字符，然后将代码和comment的每个单词以空格分开
    path = r"/home/bohong/文档/commentextract/allcomments"
    output = r"embdata.json"
    cleandata(path, output)
    # cleanfinal("data.json")

    ## 去除代码token中的停用词
    #countline("data_final.json")
    # vectorizer = CountVectorizer()
    # transformer = TfidfTransformer()
    #
    # cachedStopwords = stopwords.words("english")
    # deal_tfidf_pro("data_final.json")

