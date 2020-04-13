import pandas as pd 
from random import shuffle
from langdetect import detect

def statistics(data):
	size = data.shape
	print('Number of comments:\t\t\t', data.shape[0])

	n_no_toxic = data[ 	(data['toxic'] == 0) &
					(data['severe_toxic'] == 0) &
					(data['obscene'] == 0) &
					(data['threat'] == 0) &
					(data['insult'] == 0) &
					(data['identity_hate'] == 0) 
				].shape[0]
	print('Number of no-toxic comments:\t\t', n_no_toxic , 'percentage', 100 * n_no_toxic / size[0])
	print('Number of toxic comments:\t\t', data[data['toxic'] == 1].shape[0], '\tpercentage', 100 * data[data['toxic'] == 1].shape[0] /size[0])
	print('Number of severe_toxic comments:\t', data[data['severe_toxic'] == 1].shape[0], '\tpercentage',  100 * data[data['severe_toxic'] == 1].shape[0] / size[0])
	print('Number of obscene comments:\t\t', data[data['obscene'] == 1].shape[0], '\tpercentage', 100 * data[data['obscene'] == 1].shape[0] / size[0])
	print('Number of threat comments:\t\t', data[data['threat'] == 1].shape[0], '\tpercentage', 100 * data[data['threat'] == 1].shape[0] / size[0])
	print('Number of insult comments:\t\t', data[data['insult'] == 1].shape[0], '\tpercentage', 100 * data[data['insult'] == 1].shape[0] / size[0])
	print('Number of identity_hate comments:\t', data[data['identity_hate'] == 1].shape[0], '\tpercentage', 100 * data[data['identity_hate'] == 1].shape[0] / size[0])


def balance_on_class(data, class_name):


	data_class_yes = data[data[class_name] == 1]
	data_class_no = data[data[class_name] == 0]

	size_yes = data_class_yes.shape[0]
	size_no = data_class_no.shape[0]

	data_class_no = data_class_no.sample(n = size_yes)
	data_class = pd.concat([data_class_yes, data_class_no])

	print('TARGET CLASS:', class_name)
	print('Comments total:', data.shape[0])
	print(class_name, ' comments:', size_yes)
	print('no - ' + class_name, ' comments:', size_no)
	print('Size of balanced dataset for class ' + class_name + ':', data_class.shape[0])

	columns = ['comment_text', class_name]
	data_class = data_class[columns]
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class = data_class.sample(frac=1) #shuffle of rows
	data_class.to_csv('../datasets/' + class_name + '_balanced.csv')
	print('---------------------------------------------------------------')


'''
def keep_only_english(data):
	data_en = []
	try: 
		for i, row in data.iterrows():
			print(row['comment_text'])
			print(detect(row['comment_text']))
			if detect(row['comment_text']) == 'en':
				data_en += [{ 'comment_text' : row['comment_text'],
							  'toxic' : row['toxic'], 
							  'severe_toxic' : row['severe_toxic'], 
							  'obscene' : row['obscene'], 
							  'threat' : row['threat'], 
							  'insult' : row['insult'], 
							  'identity_hate' : row['identity_hate']
				 }]
	except Exception as e:
		print(e)

	data_en = pd.DataFrame.from_records(data_en)
	return data_en

'''

tox_train = pd.read_csv('../datasets/train.csv', encoding='utf8')
tox_test = pd.read_csv('../datasets/test.csv', encoding='utf8')
tox_test_labels = pd.read_csv('../datasets/test_labels.csv')
tox_test = tox_test.set_index('id').join(tox_test_labels.set_index('id'))
tox_test = tox_test[
					(tox_test['toxic'] != -1 ) & 
					(tox_test['severe_toxic'] != -1) & 
					(tox_test['obscene'] != -1) & 
					(tox_test['threat'] != -1) & 
					(tox_test['insult'] != -1) & 
					(tox_test['identity_hate'] != -1)
					]

data = pd.concat([tox_train, tox_test])
tox_test.to_csv('../datasets/test_with_labels.csv')
data.to_csv('../datasets/train_and_test.csv')



print('\n TRAIN DATASET STATISTICS')
statistics(tox_train)

print('\n TEST DATASET STATISTICS')
statistics(tox_test)

print('\nFULL DATASET STATISTICS')
statistics(data)


exit(1)


print('\n\n')
balance_on_class(data, 'toxic')
balance_on_class(data, 'severe_toxic')
balance_on_class(data, 'obscene')
balance_on_class(data, 'threat')
balance_on_class(data, 'insult')
balance_on_class(data, 'identity_hate')







