import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
np.random.seed(12)
num_observations = 100000
combined_dataset_file = "combined.csv"


def scikit_neural_net(train, train_label, test, test_label):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(train, train_label)	
	print 'Accuracy from sk-learn: {0}'.format(clf.score(test, test_label))


def read_data():
	dataset = pd.read_csv(combined_dataset_file, header=0)
	dataset.drop(['Body','Title','Tags','AcceptedAnswerId','CreationDate','ClosedDate','DeletionDate','Reason','Unnamed: 0','OpenOrClosed'], inplace = True, axis = 1)
	for column in dataset:
		if dataset[column].isnull().any():
			print column
	dataset['FavoriteCount'].fillna(0.0, inplace = True)
	train, test = train_test_split(dataset, test_size = 0.2)
	return train, test


if __name__ == '__main__':
	train, test = read_data()
	y = train['ReasonNumValue']
	y_test = test['ReasonNumValue']
	train.drop(['ReasonNumValue'], inplace = True, axis = 1)
	test.drop(['ReasonNumValue'], inplace = True, axis = 1)
	y_answer = y.as_matrix()
	y_answer_test = y_test.as_matrix()
	x_answer = train.as_matrix()
	x_answer_test = test.as_matrix()

	# #Creating one hot encoded vectors
	# y_train_labels = np.zeros((y_answer.shape[0], 5)).astype(int)
	# y_train_labels[np.arange(len(y_answer)), y_answer.astype(int)] = 1
	# y_test_labels = np.zeros((y_answer_test.shape[0], 5)).astype(int)
	# y_test_labels[np.arange(len(y_answer_test)), y_answer_test.astype(int)] = 1

	scikit_neural_net(x_answer, y_answer, x_answer_test, y_answer_test)

# Accuracy from sk-learn: 0.49955

