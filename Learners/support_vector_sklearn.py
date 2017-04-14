import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

combined_dataset_file = "../combined.csv"

def read_data():
	dataset = pd.read_csv(combined_dataset_file, header=0)
	dataset.drop(['Body','Title','Tags','AcceptedAnswerId','CreationDate','ClosedDate','DeletionDate','Reason','Unnamed: 0'], inplace = True, axis = 1)
	for column in dataset:
		if dataset[column].isnull().any():
			print column
	dataset['FavoriteCount'].fillna(0.0, inplace = True)
	train, test = train_test_split(dataset, test_size = 0.2)
	print dataset.head()
	return train, test


if __name__ == '__main__':
	train, test = read_data()
	y = train['OpenOrClosed']
	y_test = test['OpenOrClosed']
	train.drop(['OpenOrClosed'], inplace = True, axis = 1)
	test.drop(['OpenOrClosed'], inplace = True, axis = 1)
	y_answer = y.as_matrix()
	y_answer_test = y_test.as_matrix()
	# plot_scatter(train)
	x_answer = train.as_matrix()
	x_answer_test = test.as_matrix()

	clf = svm.SVC(gamma=0.001, C=100.)
	clf.fit(x_answer, y_answer)

	print 'Accuracy from sk-learn: {0}'.format(clf.score(x_answer_test, y_answer_test))
	#Accuracy got 0.55235