import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_scatter(dataset):
	plt.figure(figsize=(12,8))
	plt.scatter(dataset['ViewCount'],dataset['AutomatedReadingIndex'], c=colors)
	plt.xlabel("View Count")
	plt.ylabel("Readability Score")
	plt.title("Readability vs View Score")
	plt.savefig("Visualization1.png")


def read_data():
	dataset = pd.read_csv(combined_dataset_file, header=0)
	return dataset


if __name__ == '__main__':
	dataset = read_data()
	dataset.drop(['Body','Title','Tags','AcceptedAnswerId','CreationDate','ClosedDate','DeletionDate','Reason'], inplace = True, axis = 1)
	print "Values containing NaN"
	for column in dataset:
		if dataset[column].isnull().any():
			print column
	dataset['FavoriteCount'].fillna(0.0, inplace = True)
	plot_scatter(dataset)
	dataset_arr = dataset.as_matrix()
	k = 0
	for i in dataset_arr:
		if k == 5:
			break
		print i
		print ""
		print ""
		k += 1