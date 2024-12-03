import numpy as np
import joblib
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset import getClassToIdx, getData
from model import MultinomialNaiveBayes

def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

def chinese_tokenizer(text):
    return jieba.cut(text)

def process_text(X):
    return ["".join(jieba.cut(text)) for text in X]

def eval(model, X, y, n_splits):
    pass

def train(args):
    # class_list = getClassToIdx(args.data_dir)
    # print(class_list)

    # 加载数据
    X, y = getData(args.data_dir, args.max_text_cnt)
    print("数据加载完毕，开始预处理...")

    # 分词处理
    X = process_text(X)

    # 分词，计算df-idf
    print("特征提取...")
    stopwords_set = MakeWordsSet(args.stopwords_file)
    vectorizer = TfidfVectorizer(stop_words=stopwords_set, max_features=args.max_features)
    X = vectorizer.fit_transform(X)

    # 模型初始化，评估训练
    print("开始训练...")
    model = MultinomialNaiveBayes()
    eval(model, X, y, args.n_splits)

    pass

    # corpus = [
    #     "我喜欢机器学习",
    #     "自然语言处理是人工智能的一个重要方向",
    #     "深度学习在中文处理上表现优异"
    # ]
    #
    # stopwords_set = MakeWordsSet(args.stopwords_file)
    # vectorizer = TfidfVectorizer(stop_words=stopwords_set, tokenizer=chinese_tokenizer)
    #
    # X = vectorizer.fit_transform(corpus) # 转化为词袋模型
    #
    # # 查看TF-IDF矩阵
    # print(X.toarray())
