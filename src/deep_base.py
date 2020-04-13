

from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import pandas as pd
import argparse
import pickle


def str2bool(value):
    return value.lower() == 'True' or value.lower() == 'true'

def get_embeddings_name(emb_file):
	embedding_name = ''
	if emb_file == '../../resources/word2vec_toxic_300.bin':
			embedding_name = 'domain'
	elif emb_file == '../../resources/GoogleNews-vectors-negative300.bin':
		embedding_name = 'generic'
	elif emb_file == '../../resources/mimicked_Google_400k.bin':
		embedding_name = 'mimicked'

parser = argparse.ArgumentParser(description='Embeddings')
#parser.add_argument('--input-file', dest='input_file', default='../datasets/toxic_balanced.csv', type=str, action='store', help='The input file')
parser.add_argument('--target-class', dest='target_class', default='toxic', type=str, action='store', help='The target class')
parser.add_argument('--emb-file', dest='emb_file', default='../../resources/word2vec_toxic_300.bin', type=str, action='store', help='Embeddings .bin file (specify also emb size)')
parser.add_argument('--size', dest='emb_size', default=300, type=int, action='store', help='Embeddings size')
parser.add_argument('--model', dest='model', default=1, type=int, action='store', help='Select the model')
parser.add_argument('--trainable', dest='trainable', default=False, type=str2bool, action='store', help='Make embeddings trainable')
parser.add_argument('--random-embeddings', dest='random', default=False, type=str2bool, action='store', help='If true no pretrained embeddings are used')

args = parser.parse_args()

max_len = 250 		# limit of the comments lenghts inn term of number of words
num_words = 1000000 # limit of the number of words used within the model creation and evaluation


classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
n_classes = len(classes_names)
target_class = args.target_class
embedding_path = args.emb_file
embedding_dim = args.emb_size
model_type = args.model
trainable = args.trainable # if True, embeddings weights are adjusted during the training of the deep learning model 
random = args.random # if True, no pre-trained weights for the embedding layer are used
base_name = target_class + '_balanced'


tokenizer = None

for c in classes_names:
	print('TARGET CLASS:', target_class)
	for embedding_path in ['../../resources/GoogleNews-vectors-negative300.bin', '../../resources/word2vec_toxic_300.bin', '../../resources/mimicked_Google_400k.bin']
		for n_fold in range(10):
			print('> fold', n_fold)
			df_train = pd.read_csv('../train/' + base_name + '_fold_' + str(n_fold) + '.csv', encoding='utf-8').head(100)
			df_test = pd.read_csv('../test/' + base_name + '_fold_' + str(n_fold) + '.csv', encoding='utf-8').head(50)

			train_comments = []
			train_labels = []
			for i, row in df_train.iterrows():
				train_comments += [row['comment_text'].lower()]
				train_labels += [row[target_class]]

			test_comments = []
			test_labels = []
			for i, row in df_test.iterrows():
				test_comments += [row['comment_text'].lower()]
				test_labels += [row[target_class]]

			# only the first fold, in fact the data are always the same
			if tokenizer is None:
				tokenizer = Tokenizer(lower=True) 
				tokenizer.fit_on_texts(train_comments + test_comments)

			train_sequences = tokenizer.texts_to_sequences(train_comments)
			test_sequences = tokenizer.texts_to_sequences(test_comments)

			train_sequences_padded = pad_sequences(train_sequences, maxlen=max_len)
			test_sequences_padded = pad_sequences(test_sequences, maxlen=max_len)


			#Embeddings loading
			model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
			embedding_matrix = np.zeros((num_words, embedding_dim))
			for word, i in word_index.items():
				if word in model:
					embedding_vector = model[word]
					if i < num_words:
						if embedding_vector is not None:
							embedding_matrix[i] = embedding_vector
			print('> embeddings loaded. Matrix size:', embedding_matrix.shape)

			num_words = min(len(all_words) + 1, num_words)

			for model_type in range(4):

				if model_type == 0:
					model = Sequential()
					model.add(Embedding(num_words, embedding_dim, input_length=max_len))
					model.add(Dense(128))
					model.add(Dropout(0.1))
					model.add(Dense(64))
					model.add(Dropout(0.1))
					model.add(Flatten())
					model.add(Dense(1, activation='sigmoid'))
					

				elif model_type == 1:
					model = Sequential()
					model.add(Embedding(num_words, embedding_dim, input_length=max_len))
					model.add(Conv1D(128, kernel_size=10))
					model.add(Dropout(0.1))
					model.add(Conv1D(64, kernel_size=5))
					model.add(Dropout(0.1))
					model.add(Flatten())
					model.add(Dense(1, activation='sigmoid'))

				elif model_type == 2:
					model = Sequential()
					model.add(Embedding(num_words, embedding_dim, input_length=max_len))
					model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2, implementation=1))
					model.add(Dropout(0.1))
					model.add(LSTM(64, return_sequences=False, recurrent_dropout=0.2, implementation=1))
					model.add(Dropout(0.1))
					model.add(Dense(1, activation='sigmoid'))


				elif model_type == 3:
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


				model.layers[0].set_weights([embedding_matrix])
				model.layers[0].trainable = trainable
				model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
				model.fit(train_data, train_labels, epochs=20, batch_size=128, initial_epoch=0, verbose=0)
				pred_labels_scores = model.predict(test_data)
				pred_labels = np.asarray([ x[0] for x in model.predict_classes(test_data)])
				

				filename = 'predictions_model_' + str(model_type) + '_embeddings_' + get_embeddings_name(args.emb_file) + '_fold_' + str(n_fold) 
				print('> saving:', filename)
				with open('./predictions/' + filename + '.pickle', 'wb') as handle:
		    		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

				model_name = get_model_name(args)
				model.save('./models/' + model_name)












