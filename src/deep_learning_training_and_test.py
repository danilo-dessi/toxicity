import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
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
import pickle
#from attention import Attention
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

def str2bool(value):
    return value.lower() == 'True' or value.lower() == 'true'

def get_model_name(args):
	name = args.target_class
	name += '_' + str(args.model)

	if args.emb_file == '../../resources/word2vec_toxic_300.bin':
		name += '_domain'
	elif args.emb_file == '../../resources/GoogleNews-vectors-negative300.bin':
		name += '_generic'
	elif args.emb_file == '../../resources/mimicked_Google_400k.bin':
		name += '_mimicked'
	name += '.h5'
	return name


parser = argparse.ArgumentParser(description='Embeddings')
#parser.add_argument('--input-file', dest='input_file', default='../datasets/toxic_balanced.csv', type=str, action='store', help='The input file')
parser.add_argument('--target-class', dest='target_class', default='toxic', type=str, action='store', help='The target class')
parser.add_argument('--emb-file', dest='emb_file', default='../resources/word2vec_toxic_300.bin', type=str, action='store', help='Embeddings .bin file (specify also emb size)')
parser.add_argument('--size', dest='emb_size', default=300, type=int, action='store', help='Embeddings size')
parser.add_argument('--model', dest='model', default=1, type=int, action='store', help='Select the model')
parser.add_argument('--trainable', dest='trainable', default=False, type=str2bool, action='store', help='Make embeddings trainable')
parser.add_argument('--random-embeddings', dest='random', default=False, type=str2bool, action='store', help='If true no pretrained embeddings are used')

args = parser.parse_args()


max_len = 250 # limit of the comments lenghts inn term of number of words
num_words = 1000000 #limit of the number of words used within the model creation and evaluation

classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
n_classes = len(classes_names)
target_class = args.target_class
df_path = '../datasets/' + target_class + '_balanced.csv'
embedding_path = args.emb_file
embedding_dim = args.emb_size
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

print('# SAMPLE OF COMMENTS LABELS')
print(comments_labels[0:10])

#Tokenization of texts
tokenizer = Tokenizer(num_words=num_words, lower=True) 
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

embedding_name = ''
if args.emb_file == '../../resources/word2vec_toxic_300.bin':
		embedding_name = '_domain'
	elif args.emb_file == '../../resources/GoogleNews-vectors-negative300.bin':
		embedding_name = '_generic'
	elif args.emb_file == '../../resources/mimicked_Google_400k.bin':
		embedding_name = '_mimicked'
with open('./models/' + target_class + '_' + str(model_type)  + '_' + embedding_name +  '_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


word_index = tokenizer.word_index
all_words = [word for word, i in word_index.items()]

#max number of words to consider. It should be less than the number assigned to num_words as the start of the script, so that all words are considered
num_words = min(len(all_words) + 1, num_words)
most_frequent_words = [word for (word, i) in word_index.items() if i <= num_words]
all_words = most_frequent_words
print('# Word considered:', len(all_words))

data = pad_sequences(sequences, maxlen=max_len)
comments = np.asarray(comments)

#embeddings loading
model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
	if word in model:
		embedding_vector = model[word]
		if i < num_words:
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
print('# Embeddings loaded. Matrix size:', embedding_matrix.shape)


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
		model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1)))
		model.add(Dropout(0.1))
		model.add(Bidirectional(LSTM(64, return_sequences=False, recurrent_dropout=0.2, implementation=1)))
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
	model.fit(train_data, train_labels, epochs=20, batch_size=128, initial_epoch=0, verbose=0)
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
model_name = get_model_name(args)
model.save('./models/' + model_name)

print('\n# AVERAGED RESULTS: ')
print(' - acc:', mean(acc))
print(' - auc:', mean(auc))
print(' - p:', mean(p))
print(' - r:', mean(r))
print(' - f:', mean(f))








