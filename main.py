import argparse

from train import train

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--data_dir", type=str,
						default=r'F:\Compulsory Course\Data Mining\Lab2\THUCNews')
	parser.add_argument("--stopwords_file", type=str, default=r'./stopwords_cn.txt')
	parser.add_argument("--max_text_cnt", type=int, default=500)
	parser.add_argument("--n_splits", type=int, default=10)
	parser.add_argument("--max_features", type=int, default=5000)

	args = parser.parse_args()

	# train(args)
	train(args)