import mxnet as mx
from bert_embedding import BertEmbedding
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, GaussianNoise, Activation, MaxoutDense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
import sys
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout, Bidirectional, LSTM, GaussianNoise, Activation, MaxoutDense
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.metrics import confusion_matrix
from keras.metrics import binary_accuracy
import argparse
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import datetime
import pickle
import gc


ctx = mx.gpu(0)
print(ctx)


def str2bool(value):
    return value.lower() == 'True' or value.lower() == 'true'


parser = argparse.ArgumentParser(description='Embeddings')
#parser.add_argument('--input-file', dest='input_file', default='../datasets/toxic_balanced.csv', type=str, action='store', help='The input file')
parser.add_argument('--target-class', dest='target_class', default='toxic', type=str, action='store', help='The target class')
parser.add_argument('--model', dest='model', default=1, type=int, action='store', help='Select the model')
parser.add_argument('--trainable', dest='trainable', default=False, type=str2bool, action='store', help='Make embeddings trainable')
parser.add_argument('--random-embeddings', dest='random', default=False, type=str2bool, action='store', help='If true no pretrained embeddings are used')

args = parser.parse_args()

max_len = 250 # limit of the comments lenghts in term of number of words
classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
n_classes = len(classes_names)
target_class = args.target_class
df_path = '../datasets/' + target_class + '_balanced.csv'
model_type = args.model
trainable = args.trainable # if True, embeddings weights are adjusted during the training of the deep learning model 
random = args.random # if True, no pre-trained weights for the embedding layer are used
print('TARGET CLASS:', target_class)


df = pd.read_csv(df_path, encoding='utf-8')
size = df.shape
print('Number of comments:', size[0])
df = df.sample(frac=1) #shuffle of rows
df = df.sample(frac=1) #shuffle of rows

comments = []
comments_labels = []
for i, row in df.iterrows():
	comments += [row['comment_text'].lower()]
	labels = []
	comments_labels += [row[target_class]]	

comments_labels = np.asarray(comments_labels)

print('BERT START', str(datetime.datetime.now()))
bert_embedding = BertEmbedding(ctx=ctx, model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
result = bert_embedding(comments)
gc.collect()
print('BERT END', str(datetime.datetime.now()))

id2emd = {}
id2word = {}
id_n = 1
embedding_dim = 0

sequences = []

for (vocab_list, emb_list) in result:
	sequence = []
	for i in range(len(vocab_list)):

		if embedding_dim == 0:
			embedding_dim = len(emb_list[i])
		sequence += [id_n]
		id2emd[id_n] = emb_list[i]
		id2word[id_n] = vocab_list[i]
		id_n += 1
	sequences += [sequence]

data = pad_sequences(sequences, maxlen=max_len)
comments = np.asarray(comments)

keys = sorted(id2word.keys())
embedding_matrix = np.zeros((id_n, embedding_dim))
for id_key in keys:
	embedding_vector = id2emd[id_key]
	embedding_matrix[id_key] = embedding_vector
print('# Embeddings loaded >> Matrix size:', embedding_matrix.shape)



with open('./models/'+ target_class + '_' + str(model_type) + '_id2emb.pickle', 'wb') as handle:
    pickle.dump(id2emd, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
word2id = {}
for id in id2word:
	word = id2word[id]
	word2id[word] = id
with open('./models/' + target_class + '_' + str(model_type) +  '_word2id.pickle', 'wb') as handle:
    pickle.dump(word2id, handle, protocol=pickle.HIGHEST_PROTOCOL)



print('BERT LOADED', str(datetime.datetime.now()))
num_words = id_n

cv = KFold(n_splits=10, shuffle=True)
n_fold = 0
acc = []
auc = []
p = []
r = []
f = []
model = None
for train_index, test_index in cv.split(data):
	n_fold += 1
	
	train_data = data[train_index]
	train_labels = comments_labels[train_index]
	test_data = data[test_index]
	test_labels = comments_labels[test_index]

	s = np.arange(train_data.shape[0])
	np.random.shuffle(s)
	train_data = train_data[s]
	train_labels = train_labels[s]
	
	if model_type == 1:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(Dense(128))
		model.add(Dropout(0.1))
		model.add(Dense(64))
		model.add(Dropout(0.1))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		

	elif model_type == 2:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(Conv1D(128, kernel_size=10))
		model.add(Dropout(0.1))
		model.add(Conv1D(64, kernel_size=5))
		model.add(Dropout(0.1))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

	elif model_type == 3:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1))
		model.add(Dropout(0.1))
		model.add(LSTM(64, return_sequences=False, recurrent_dropout=0.2, implementation=1))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))

	elif model_type == 4:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1))
		model.add(Dropout(0.1))
		model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.2, implementation=1))
		model.add(SeqWeightedAttention())
		model.add(Dense(1, activation='sigmoid'))

	elif model_type == 5:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1)))
		model.add(Dropout(0.1))
		model.add(Bidirectional(LSTM(64, return_sequences=False, recurrent_dropout=0.2, implementation=1)))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))

	elif model_type == 6:
		model = Sequential()
		model.add(Embedding(num_words, embedding_dim, input_length=max_len))
		model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1)))
		model.add(Dropout(0.1))
		model.add(Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.2, implementation=1)))
		model.add(Dropout(0.1))
		model.add(Dense(64))
		model.add(Dropout(0.1))
		model.add(Dense(1, activation='sigmoid'))
	else:
		print('No module has been chosen')
		exit(1)

	if not random:
		model.layers[0].set_weights([embedding_matrix])
	
	model.layers[0].trainable = trainable

	if n_fold == 1:
		print(model.summary())
	
	
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
	model.fit(train_data, train_labels, epochs=20, batch_size=128, initial_epoch=0, verbose=0	)
	pred_labels_scores = model.predict(test_data)
	pred_labels = np.asarray([ x[0] for x in model.predict_classes(test_data) ])
	
	
	fpr, tpr, thresholds = roc_curve(test_labels, pred_labels_scores, pos_label=None)
	auc_value = metrics.auc(fpr, tpr)
	acc_binary = accuracy_score(test_labels, pred_labels)
	p_binary = precision_score(test_labels, pred_labels)
	r_binary = recall_score(test_labels, pred_labels)
	f_binary = f1_score(test_labels, pred_labels)
	
	acc += [acc_binary]
	auc += [auc_value]
	p += [p_binary]
	r += [r_binary]
	f += [f_binary]
 
	print('\n# N Fold', n_fold)
	print('target class:', target_class)
	print('acc:', acc_binary)
	print('auc:', auc_value)
	print('p:', p_binary)
	print('r:', r_binary)
	print('f:', f_binary)

#Last model save
model_name = get_model_name(target_class + '_' + str(model_type) + '_bert.h5')
model.save('./models/' + model_name)

print('\n# AVERAGED RESULTS: ')
print(' - acc:', mean(acc))
print(' - auc:', mean(auc))
print(' - p:', mean(p))
print(' - r:', mean(r))
print(' - f:', mean(f))
















