import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


np.random.seed(12)
num_observations = 100000
combined_dataset_file = "combined.csv"
colors = ['red','blue']

def sigmoid(scores):
	return 1/(1+np.exp(-scores))

def log_likelihood(features, target, weights):
	scores = np.dot(features, weights)
	ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
	return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept):
	if add_intercept:
		intercept = np.ones((features.shape[0], 1))
		features = np.hstack((intercept, features))

	weights = np.zeros(features.shape[1])
    
	for step in xrange(num_steps):
		scores = np.dot(features, weights)
		predictions = sigmoid(scores)

        # Update weights with gradient
		output_error_signal = target - predictions
		gradient = np.dot(features.T, output_error_signal)
		weights += learning_rate * gradient
        
        # Print log-likelihood every so often
		if step % 10000 == 0:
			print "Congrats! You have completed " + str(step) + " steps"
        
	return weights


def plot_scatter(dataset):
	plt.figure(figsize=(12,8))
	plt.scatter(dataset['ViewCount'],dataset['AutomatedReadingIndex'], c=colors)
	plt.xlabel("View Count")
	plt.ylabel("Readability Score")
	plt.title("Readability vs View Score")
	plt.savefig("Visualization1.png")


def read_data():
	dataset = pd.read_csv(combined_dataset_file, header=0)
	dataset.drop(['Body','Title','Tags','AcceptedAnswerId','CreationDate','ClosedDate','DeletionDate','Reason','Unnamed: 0'], inplace = True, axis = 1)
	for column in dataset:
		if dataset[column].isnull().any():
			print column
	dataset['FavoriteCount'].fillna(0.0, inplace = True)
	train, test = train_test_split(dataset, test_size = 0.2)
	return train, test

# Main function for Logistic Regression
if __name__ == '__main__':
	train, test = read_data()
	y = train['OpenOrClosed']
	y_test = test['OpenOrClosed']
	train.drop(['OpenOrClosed'], inplace = True, axis = 1)
	test.drop(['OpenOrClosed'], inplace = True, axis = 1)
	y_answer = y.as_matrix()
	y_answer_test = y_test.as_matrix()
	plot_scatter(train)
	x_answer = train.as_matrix()
	x_answer_test = test.as_matrix()
	
	weights = weights = logistic_regression(x_answer, y_answer,
                     num_steps = 30000, learning_rate = 5e-5, add_intercept=True)
	print "Crude Logistic Regression:"
	print weights

	#For Validation I am checking whether the weight computed by my code is same as what sklearn optimized version gives me.
	clf = LogisticRegression(fit_intercept=True, C = 1e15)
	clf.fit(x_answer, y_answer)

	print "SkLearn Logistic Regression answer:"
	print clf.intercept_, clf.coef_

	test_intercept = True
	#Checking accuracy
	#Here the predictions are made using our code written from scratch
	if test_intercept is True:
		intercept = np.ones((x_answer_test.shape[0], 1))
		x_answer_test_final = np.hstack((intercept, x_answer_test))
	final_scores = np.dot(x_answer_test_final, weights)
	preds = np.round(sigmoid(final_scores))

	print 'Accuracy from scratch: {0}'.format((preds == y_answer_test).sum().astype(float) / len(preds))
	print 'Accuracy from sk-learn: {0}'.format(clf.score(x_answer_test, y_answer_test))