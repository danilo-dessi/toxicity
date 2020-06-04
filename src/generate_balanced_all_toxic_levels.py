import pandas as pd

def labels2vec(labels):
	vec = []
	toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	for t in toxicity_labels:
		if t in labels:
			vec += [1]
		else:
			vec += [0]

balanced_dataset_paths = [
	'../datasets/toxic_balanced.csv',
	'../datasets/severe_toxic_balanced.csv',
	'../datasets/obscene_balanced.csv',
	'../datasets/threat_balanced.csv',
	'../datasets/insult_balanced.csv',
	'../datasets/identity_hate_balanced.csv'
]

less_representative_number = 689

comments = {}
toxicity_type = []
for dataset in balanced_dataset_paths:
	df = pd.read_csv(dataset)
	toxicity = dataset.replace('../datasets/', '').replace('_balanced.csv','')
	for i,r in df.iterrows():
		if r[toxicity] == '1':
			if r['comment_text'] not in comments:
				comments[r['comment_text']] = []
			comments[r['comment_text']] += [r[toxicity]]

comments_vectors = [(c, labels2vec[c]) for c in comments]
comments = [c for (c, vec) in comments_vectors]
vecs = [vec for (c, vec) in comments_vectors]




















