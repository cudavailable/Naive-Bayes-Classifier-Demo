import os

def getClassToIdx(data_dir):
	"""
	输入：数据集的绝对路径
	返回：数据集的类别列表
	"""
	class_list = []
	print(data_dir)
	for category in os.listdir(data_dir):
		category_path = os.path.join(data_dir, category)
		if os.path.isdir(category_path):
			class_list.append(category)

	return class_list

def getData(data_dir, max_text_cnt):
	"""
	输入：r'F:\Compulsory Course\Data Mining\Lab2\THUCNews'
	返回：文本数据X和对应的标签向量y
	"""
	X, y = [], []
	for category in os.listdir(data_dir):
		category_path = os.path.join(data_dir, category)
		if os.path.isdir(category_path):
			"""e.g. /data/体育 文件夹存在"""
			text_cnt = 0
			for text in os.listdir(category_path):
				text_path = os.path.join(category_path, text)
				"""e.g. /data/体育/101003.txt"""
				if os.path.isfile(text_path):
					with open(text_path, "r", encoding='utf-8') as file:
						X.append(file.read().strip())
						y.append(category)
						text_cnt += 1

				if max_text_cnt is not None and text_cnt >= max_text_cnt:
					break

	return X, y