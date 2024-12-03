import numpy as np
from collections import defaultdict

class MultinomialNaiveBayes:
	"""
	多项式朴素贝叶斯模型实现
	"""
	def __init__(self):
		self.classes = None # 去重后的类标签序列
		self.voc_size = None # 词集长度
		self.prior = None # P(y)
		self.cond = None # P(X|y)

	def fit(self, X, y):
		"""
		训练多项式朴素贝叶斯分类器
		:param X: 训练文本数据矩阵 (n_samples, n_features)
		:param y: 训练数据标签向量 (n_samples, )
		:return: None
		"""
		self.classes = np.unique(y) # 去重类标签序列
		n_classes = len(self.classes) # 统计类数

		self.voc_size = X.shape[1] # 特征个数即词集长度
		self.cond = defaultdict(lambda: defaultdict(float))

		# 计算先验分布 P(y)
		self.prior = {}
		for c in self.classes:
			self.prior[c] = np.sum(y==c) / len(y) # P(y)

		# 计算后验分布 P(xi|y)
		for c in self.classes:
			X_c = X[y==c]
			# x_c = np.sum(X_c, dim=1)
			class_word_count = X_c.sum(axis=0) # 某类主题对应的各词出现频数
			total_word_count = class_word_count.sum()
			for i in range(self.voc_size):
				self.cond[c][i] = (class_word_count[0, i] + 1) / (total_word_count + self.voc_size)

	def predict(self, X):
		"""
		输入文本数据进行主题预测
		:param X: 新闻文本数据矩阵  (n_samples, n_features)
		:return: predictions: 相应类别向量 (n_samples, )
		"""
		predictions = []
		for x in X:
			post = {}
			for c in self.classes:
				# P(c|X) -> P(c) * product(P(xi|c))
				# we calculate logP(c|X) here
				log_prob = np.log(self.prior[c])
				for i in range(self.voc_size):
					if x[i] > 0:
						log_prob += x[i] * np.log(self.cond[c][i])
				post[c] = log_prob # P(c|X)

			predictions.append(max(post, key=post.get())) # 选择对数概率值最大的类别

		return predictions
