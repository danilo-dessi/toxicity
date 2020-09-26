# Toxicity detection

This repository contains the scripts developed to perform the analysis reported in the paper: 

```
An Assessment of Deep Learning Models and Word Embeddings for Toxicity Detection 
within Online Textual Comments authored by Danilo Dessi', Diego Reforgiato Recupero, and Harald Sack.
``` 

If you have any question plese contact: [Danilo Dessi'](mailto:danilo.dessi@kit.edu)


## Repository Description

- **src/** contains all the scripts that have been used.
	- *deep_learning_training_and_test.py* is the script that uses Word2Vec embeddings.
	- *deep_learning_training_and_test_BERT.py* is the script that uses BERT embeddings.
	- *ml.py* is the script used for our baselines.
	- *preprocess_toxic_kaggle.py* is the script used to create the balanced datasets.
	- *generate_toxicity_embeddings.py* is the script used to create domain-based word embeddings.
	- *\*.sh* are the scripts used to automatically perform all our experiments.  
	- *results/* contains the logs of the execution of our scripts.
- **datasets/** contains the dataset used for this study.
	- *train.csv* the original training set of the dataset (see [Kaggle data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)).
	- *test.csv* the original test set of the dataset (see [Kaggle data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)).
	- *test_labels.csv* the original labels of the dataset (see [Kaggle data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)).
	- *test_with_labels.csv* the file that merge the test data wih the corresponding labels.
	- *X_balanced* the balanced dataset for the class X.
- **resources/** should contain the following word embedding models:
	- *word2vec_toxic_300.bin* the word embeddings trained on the domain.
	- *mimicked_Google_400k.bin* the mimicked word embeddings containing all words of our domain, plus other coming from the pre-trained model for a maximum of 400k embeddings.
	- *GoogleNews-vectors-negative300.bin* the Google pre-trained embeddings available [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

Due to upload limits of the files size on github, contact [Danilo Dessi'](mailto:danilo.dessi@kit.edu) to get *word2vec_toxic_300.bin* and *mimicked_Google_400k.bin* models.

If you have any further question, feel free to contact us!

