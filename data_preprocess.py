# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
from time import time
import data_plot
import pdb 

def loadData(fileName):
    df=pd.read_csv(fileName)
    
    # fill NaN if any
    if df.isnull().values.any():
        imputer=KNNImputer(n_neighbors=5)
        df=imputer.fit_transform(df)
    
    
    # rescale, define X, y
    df_x=df.drop(['filename','length','label'],axis=1)
    scaler=MinMaxScaler()
    X=scaler.fit_transform(df_x)
    
    df_y=df['label']
    y=df_y.to_numpy()
    
    return X, y


# visualize the data by dimension reduction
def data_visualize(X, y, method='all', n_comp=2):
    n_class=len(np.unique(y))

    Methods={
    'PCA' : PCA(n_components=n_comp),
    'LDA' : LinearDiscriminantAnalysis(n_components=n_comp),
    'MDS' : manifold.MDS(n_components=n_comp, max_iter=200, n_init=1),
    'tSNE' : manifold.TSNE(n_components=n_comp, init='pca', random_state=0),
    'ICA' : FastICA(n_components=n_comp,random_state=0)
    }

    if method=='all':
        method=list(Methods.keys())
        
    for i, (label,reduceDimen) in enumerate(Methods.items()):
        #pdb.set_trace()
        if label in method:
            t0=time()
            if label=='LDA':
                Y=reduceDimen.fit_transform(X,y)
            else:      
                Y=reduceDimen.fit_transform(X)
            t1=time()
            print("%s: %.2g sec" % (label, t1 - t0))
            data_plot.plot_embedding(Y, y,label)



