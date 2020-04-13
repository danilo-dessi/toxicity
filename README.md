# Toxicity detection

This repository contains the scripts developed to perform the analysis reported in the paper: 

```
An Assessment of Deep Learning Models and Word Embeddings for Toxicity Detection within Online Textual Comments authored by Danilo Dessi' and Diego Reforgiato Recupero.
``` 

If you have any question plese contact: [Danilo Dessi'](mailto:danilo_dessi@unica.it)


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
- **resources/** contains the word embeddings.
