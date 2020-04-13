import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from gensim.models.keyedvectors import KeyedVectors
import sys
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.metrics import confusion_matrix
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def str2bool(value):
    return value.lower() == 'True' or value.lower() == 'true'

def get_model_name(args):
	name = args.target_class
	name += '_model' + str(args.model)

	if args.emb_file == '../../resources/word2vec_toxic_300.bin':
		name += '_domain'
	elif args.emb_file == '../../resources/GoogleNews-vectors-negative300.bin':
		name += '_generic'
	name += '.h5'
	return name


parser = argparse.ArgumentParser(description='Embeddings')
parser.add_argument('--target-class', dest='target_class', default='toxic', type=str, action='store', help='The target class')
parser.add_argument('--ml-algorithm', dest='ml', default='svm', type=str, action='store', help='The Machine Learning algorithm')
args = parser.parse_args()

max_len = 250 # limit of the comments lenghts inn term of number of words
num_words = 1000000 #limit of the number of words used within the model creation and evaluation

classes_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
n_classes = len(classes_names)
target_class = args.target_class
df_path = '../datasets/' + target_class + '_balanced.csv'
ml_algorithm = args.ml

print('TARGET CLASS:', target_class)
print('ML ALGORITHM:', ml_algorithm)


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

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(comments)
# summarize encoded vector
print(tfidf_matrix.shape)
print(tfidf_matrix[0])



cv = KFold(n_splits=10, shuffle=True)
n_fold = 0
acc = []
auc = []
p = []
r = []
f = []
model = None
for train_index, test_index in cv.split(tfidf_matrix):
	n_fold += 1
	
	train_data = tfidf_matrix[train_index]
	train_labels = comments_labels[train_index]
	test_data = tfidf_matrix[test_index]
	test_labels = comments_labels[test_index]

	print('Shape training data:', train_data.shape)
	print('Shape test data:', test_data.shape)
	s = np.arange(train_data.shape[0])
	np.random.shuffle(s)
	train_data = train_data[s]
	train_labels = train_labels[s]
	
	if ml_algorithm.lower() == 'svm':
		clf = SVC()
	elif ml_algorithm.lower() == 'dt':
		clf = DecisionTreeClassifier()
	elif ml_algorithm.lower() == 'nb':
		clf = GaussianNB()
	elif ml_algorithm.lower() == 'rf':
		clf = RandomForestClassifier(n_estimators=100)
	elif ml_algorithm.lower() == 'mlp':
		clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
	else:
		print('ERROR: the chosen classification algorithm is not available. Please use: cvm, dt, nb, rf, mlp')
		exit(1)

	clf.fit(train_data, train_labels)
	pred_labels = clf.predict(test_data)
	
	
	acc_binary = accuracy_score(test_labels, pred_labels)
	p_binary = precision_score(test_labels, pred_labels)
	r_binary = recall_score(test_labels, pred_labels)
	f_binary = f1_score(test_labels, pred_labels)
	
	acc += [acc_binary]
	p += [p_binary]
	r += [r_binary]
	f += [f_binary]
 
	print('\n# N Fold', n_fold)
	print('target class:', target_class)
	print('acc:', acc_binary)
	print('p:', p_binary)
	print('r:', r_binary)
	print('f:', f_binary)
	print(pred_labels[0:10])
	print(test_labels[0:10])

print('\n# AVERAGED RESULTS: ')
print(' - acc:', mean(acc))	
print(' - p:', mean(p))
print(' - r:', mean(r))
print(' - f:', mean(f))








