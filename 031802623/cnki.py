# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import jieba
from gensim import  corpora,models,similarities
import codecs
import jieba.posseg as pseg

def read_stop_word(file_path):
    file = file_path
    stopwords = codecs.open(file,'r',encoding='utf8').readlines()
    stopwords = [ w.strip() for w in stopwords ]
    return stopwords

stopwords = read_stop_word("stop_word.txt")

stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r']

def create_file(file_path,msg):
    f=open(file_path,"a")
    f.write(msg)
    f.close

def tokenization(filename):
    result = []
    with open(filename, 'r',encoding='utf8') as f:
        text = f.read()
        words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

def cut_words(file):
    with open(file, 'r',encoding="utf-8") as f:
        text = f.read()
        words = jieba.lcut(text)
    return words

def drop_Disable_Words(cut_res,stopwords):
    res = []
    for word in cut_res:
        if word in stopwords or word =="\n" or word =="\u3000":
            continue
        res.append(word)
    return res


def mygensim(testpath,copypath,respath):
    files = [testpath,copypath]

    corpus = []
    for file in files:
        #分词
        cut_res = cut_words(file)
        #去停用词
        res = drop_Disable_Words(cut_res,stopwords)
        corpus.append(res)


    dictionary = corpora.Dictionary(corpus)
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    
    query = tokenization(testpath)
    query_bow = dictionary.doc2bow(query)
    
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]

    #使用TF-IDF模型计算相似度
    TF_IDF(tfidf_vectors,query_bow,respath)

#建立TF-IDF模型
def TF_IDF(tfidf_vectors,query_bow,respath):
    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_bow]
    create_file(respath,"TF-IDF模型计算结果为："+str(1-float(list(enumerate(sims))[0][1])))

testpath = input('输入文本1路径：')
copypath = input('输入文本2路径：')
respath = input('答案文件路径：')
mygensim(testpath,copypath,respath)
