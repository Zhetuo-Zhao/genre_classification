# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 21:56:29 2020

@author: zzhao
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pdb


def plot_embedding(X, y,title=None):
#   x_min, x_max = np.min(X, 0), np.max(X, 0)
#    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    classes=np.unique(y)
    for i in range(X.shape[0]):
        #pdb.set_trace()
        plt.text(X[i, 0], X[i, 1], str(y[i][:2]),
                 color=plt.cm.Set1(int(np.where(classes==y[i])[0]) / len(classes)),
                 fontdict={'weight': 'bold', 'size': 9})
    
        
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
