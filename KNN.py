# %%

import data_preprocess
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np
from time import time

def evaluate_accuracy(predict, test):
    correct_proportion=np.sum(predict==test)/len(test)
    return correct_proportion


X, y=data_preprocess.loadData('../Data/features_30_sec.csv')

# Split the data into training and testing sets
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng)

y_predicted={}
# KNN
n_neighbors = 5;
for weights in ['uniform', 'distance']:
    t0=time()
    KNN = KNeighborsClassifier(n_neighbors, weights=weights)
    KNN.fit(X_train, y_train)
    t1=time()
    _=KNN.predict(X_test)
    y_predicted['KNN_' + weights]=_
    print('KNN, takes %.2g sec, achieves acurray %.2g' %(t1-t0,evaluate_accuracy(_,y_test)))