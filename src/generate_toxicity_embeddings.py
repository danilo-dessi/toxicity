from nltk.tokenize import RegexpTokenizer
import pandas as pd
import argparse
import gensim
import os
import nltk


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Embeddings')
	parser.add_argument('--input-file', dest='input_file', default='../datasets/train_and_test.csv', type=str, action='store', help='The input file')
	parser.add_argument('--size', dest='emb_size', default=300, type=int, action='store', help='Embeddings size')
	parser.add_argument('--iter', dest='iter', default=10, type=int, action='store', help='Number of iterations the process will execute')
	parser.add_argument('--workers', dest='workers', default=4, type=int, action='store', help='Number of threads the process will use')
	parser.add_argument('--min-count', dest='min_count', default=1, type=int, action='store', help='Minimum number of occurrencies each word must have')
	parser.add_argument('--window', dest='window', default=5, type=str, action='store', help='Size of the window')
	args = parser.parse_args()

	print ('# Preparing text for Word2Vec algorithm')
	texts = []
	data = pd.read_csv(str(args.input_file))
	tokenizer = RegexpTokenizer(r'\w+')
	texts = [tokenizer.tokenize(row['comment_text'].lower()) for i, row in data.iterrows() ]


	print ('# Word2Vec embeddings generation')
	model = gensim.models.Word2Vec(texts, size=args.emb_size, window=args.window, min_count=args.min_count, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format('../resources/word2vec_toxic_'+ str(args.emb_size) + '.txt', binary=False)
	model.wv.save_word2vec_format('../resources/word2vec_toxic_'+ str(args.emb_size) + '.bin', binary=True)
	print ('File ' + 'word2vec_toxic_'+ str(args.emb_size) + '.bin generated')





