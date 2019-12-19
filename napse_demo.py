#!/usr/bin/env python3.7

import os
#import cv2
import numpy as np
import scipy
import sklearn
from sklearn import datasets
from napse import *
from sklearn.preprocessing import normalize
from initializers import *



def load_dataset(DataNoise = 0.05, Visualize = False):
    #np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=DataNoise)
    #np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=DataNoise)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))

    return train_X, train_Y, test_X, test_Y


np.random.seed(1001)
train_X, train_Y, test_X, test_Y = load_dataset(DataNoise = 0.15, Visualize=False)

def filter_func (x_):
    return sklearn.preprocessing.normalize(x_)
def filter_dummy(x_):
    return x_

nn =      Layer("input_layer",(2,1)) \
            | RandomInitializer()\
        > Layer("h1", (5,1)) \
            | PreFilter("Filter1", filter_dummy)\
            | PostFilter("Filter2", filter_dummy)\
        > Layer("output_layer",(1,1), activation=Activation.SIGMOID)\
        > CostLayer()\
            | SGD(batch_size=32)\
            | L2Regularizer(lambda_=1e-4)

nn.train(train_X,train_Y, epochs=15000, learning_rate=0.2)
train_predict = nn.predict(test_X)
train_predict[train_predict>=0.5]=1
train_predict[train_predict<0.5]=0
print("train accuracy: {} ".format(np.mean(train_predict == test_Y)) )


#print (nn.hidden_layers)
#
#print(nn.input_layer.next.A().shape)
#print(nn.input_layer.next.A())
#
#print(nn.input_layer.next.next.A().shape)
#print(nn.input_layer.next.next.A())

