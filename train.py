import os.path

import numpy as np
import joblib
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from dataset import getClassToIdx, getData
from model import MultinomialNaiveBayes
from logger import Logger

def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip()
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

def process_text(X):
    return ["".join(jieba.cut(text)) for text in X]

def eval(model, X, y, n_splits):
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        "accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
        "precision_micro": [],
        "recall_micro": [],
        "f1_micro": [],
    }
    y = np.array(y)  # 确保 y 是 NumPy 数组

    # 根据10折标准划分训练集和测试集
    for train_idx, test_idx in kf.split(X, y):
        train_idx = np.array(train_idx, dtype=int)  # 确保是整数数组
        test_idx = np.array(test_idx, dtype=int)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision_macro"].append(precision_score(y_test, y_pred, average="macro"))
        metrics["recall_macro"].append(recall_score(y_test, y_pred, average="macro"))
        metrics["f1_macro"].append(f1_score(y_test, y_pred, average="macro"))
        metrics["precision_micro"].append(precision_score(y_test, y_pred, average="micro"))
        metrics["recall_micro"].append(recall_score(y_test, y_pred, average="micro"))
        metrics["f1_micro"].append(f1_score(y_test, y_pred, average="micro"))

        # 计算平均值
        for key in metrics:
            metrics[key] = sum(metrics[key]) / len(metrics[key])
        return metrics

def train(args):
    # class_list = getClassToIdx(args.data_dir)
    # print(class_list)
    # 保存日志
    if args.log_dir is not None and not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'))

    logger.write("实验配置：\n---------------------\n")
    logger.write(f"每类新闻采用文本数：{args.max_text_cnt}\n")
    logger.write(f"文本特征向量的最大长度：{args.max_features}\n")

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
    metrics = eval(model, X, y, args.n_splits)

    logger.write("\n实验结果：\n---------------------\n")
    logger.write(f"Accuracy: {metrics['accuracy']:.6f} \n")
    logger.write(f"Precision (Macro): {metrics['precision_macro']:.6f} \n")
    logger.write(f"Recall (Macro): {metrics['recall_macro']:.6f} \n")
    logger.write(f"F1-Value (Macro): {metrics['f1_macro']:.6f} \n")
    logger.write(f"Precision (Micro): {metrics['precision_micro']:.6f} \n")
    logger.write(f"Recall (Micro): {metrics['recall_micro']:.6f} \n")
    logger.write(f"F1-Value (Micro): {metrics['f1_micro']:.6f} \n")
