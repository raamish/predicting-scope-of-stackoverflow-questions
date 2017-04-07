import pandas as pd
import numpy as np
import os

train_open_file = "open-questions.csv"
train_close_file = "closed-questions.csv"

#Importing Dataset For Open Questions
dataset_open = pd.read_csv(train_open_file ,header = 0)
dataset_open['Reason'] = "open"
dataset_open.drop(['PostId','OwnerUserId'], inplace = True, axis = 1, errors='ignore')

#Importing Dataset For Closed Questions
dataset_closed = pd.read_csv(train_close_file, header = 0)
dataset_closed.rename(columns={"Reputation":"OwnerReputation", "UpVotes":"OwnerUpVoteCount", "Name":"Reason"}, inplace = True)
dataset_closed.drop(['PostId','OwnerUserId','Id','Comment','Description'], inplace = True, axis = 1, errors='ignore')

print "Open Dataset Overview"
print dataset_open.columns.values

print "Closed Dataset Overview"
print dataset_closed.columns.values

#Combining Datasets
combined_dataset = dataset_open.append(dataset_closed)

print combined_dataset.head(5)
print combined_dataset.shape
combined_dataset.to_csv('combined.csv')