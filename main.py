# %%

import data_preprocess
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from time import time
from tensorflow import keras
import deep_NN
import pdb

def evaluate_accuracy(predict, test):
    correct_proportion=np.sum(predict==test)/len(test)
    return correct_proportion

# %%
X, y=data_preprocess.loadData()
n_classes=len(np.unique(y))

# Split the data into training and testing sets
rng = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng)


y_predicted={}


# %% KNN
for weights in ['uniform', 'distance']:
    t0=time()
    KNN = KNeighborsClassifier(int(n_classes), weights=weights)
    KNN.fit(X_train, y_train)
    t1=time()
    _test=KNN.predict(X_test)
    y_predicted['KNN_' + weights]=_test
    _train=KNN.predict(X_train)
  
    print('KNN %s, takes %.2g sec, achieves accuracy test %.2g, train %.2g' 
          %(weights,t1-t0,evaluate_accuracy(_test,y_test),evaluate_accuracy(_train,y_train)))
    

# %% SVM
C=1
clf = svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
clf.fit(X_train,y_train)
_test=clf.predict(X_test)
_train=clf.predict(X_train)
print('svm achieves accuracy test %.2g, train %.2g' 
          %(evaluate_accuracy(_test,y_test),evaluate_accuracy(_train,y_train)))
    
# %% LSTM
inputDim4RNN, time_steps=57, 10
_=np.reshape(X_train,[-1,inputDim4RNN,time_steps])
X_train_4RNN=np.transpose(_,(0,2,1))
_=np.reshape(X_test,[-1,inputDim4RNN,time_steps])
X_test_4RNN=np.transpose(_,(0,2,1))

batch_size=100
units=128
verbose=0
epochs=50
LSTM = deep_NN.build_model_LSTM(units,inputDim4RNN,n_classes)
opt = keras.optimizers.SGD(learning_rate=0.005, momentum=0.6)
LSTM.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=opt,    metrics=["accuracy"],
)

LSTM.fit(X_train_4RNN, y_train, validation_data=(X_test_4RNN, y_test),
         batch_size=batch_size, epochs=epochs,verbose=verbose)

results_LSTM = LSTM.evaluate(X_test_4RNN, y_test)


# RNN
RNN = deep_NN.build_model_RNN(units,inputDim4RNN,n_classes)

RNN.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=opt,    metrics=["accuracy"],
)

RNN.fit(X_train_4RNN, y_train, validation_data=(X_test_4RNN, y_test),
         batch_size=batch_size, epochs=epochs,verbose=verbose)
results_RNN = RNN.evaluate(X_test_4RNN, y_test)



# %% NN
inputDim=570
NN = deep_NN.build_model_NN(units,inputDim,n_classes)
NN.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="Adam",    metrics=["accuracy"],
)

NN.fit(X_train, y_train, validation_data=(X_test, y_test),
         batch_size=batch_size, epochs=200,verbose=0)
results_NN = NN.evaluate(X_test, y_test)



