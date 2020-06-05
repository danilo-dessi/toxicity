import pandas as pd

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def labels2vec(labels):
	vec = []
	for t in toxicity_labels:
		if t in labels:
			vec += [1]
		else:
			vec += [0]
	return vec

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

	c = 0
	for i,r in df.iterrows():
		if r[toxicity] == 1:

			if r['comment_text'] not in comments:
				comments[r['comment_text']] = []
			comments[r['comment_text']] += [toxicity]

			c += 1
			if c == less_representative_number:
				break


comments_vectors = [(c, labels2vec(comments[c])) for c in comments]
comments = [c for (c, vec) in comments_vectors]
vecs = [vec for (c, vec) in comments_vectors]


toxic_column = [v[0] for v in vecs]
severe_toxic_column = [v[1] for v in vecs]
obscene_column = [v[2] for v in vecs]
threat_column = [v[3] for v in vecs]
insult_column = [v[4] for v in vecs]
identity_hate_column = [v[5] for v in vecs]


data = {
	'comment_text' : comments,
	'toxic' : toxic_column,
	'severe_toxic' : severe_toxic_column,
	'obscene' : obscene_column,
	'threat' : threat_column,
	'insult' : insult_column,
	'identity_hate': identity_hate_column
}


df = pd.DataFrame.from_dict(data)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)
print('Size of all classes balanced dataset:', df.shape)
df.to_csv('../datasets/all_classes_balanced.csv')























