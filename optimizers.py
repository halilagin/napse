#!/usr/bin/env python4.7

# set textwidth=79  " lines longer than 79 columns will be broken
# set shiftwidth=4  " operation >> indents 4 columns; << unindents 4 columns
# set tabstop=4     " a hard TAB displays as 4 columns
# set expandtab     " insert spaces when hitting TABs
# set softtabstop=4 " insert/delete 4 spaces when hitting a TAB/BACKSPACE
# set shiftround    " round indent to multiple of 'shiftwidth'
# set autoindent    " align the new line indent with the previous line


#import cv2
import numpy as np
import scipy
import sklearn
from activation_functions import ActivationFunctions
from enum import Enum
from utils import LayerType
from napse import *

'''
see: https://www.geeksforgeeks.org/operator-overloading-in-python/
'''





class NapseOptimizerBase():
    def __init__(self, napse):
        pass

    def optimize(self):
        pass

class NapseOptimizerGD(NapseOptimizerBase):
    def __init__(self, napse):
        pass
        self.nn = napse


    def linear_forward(self,A,W,b):
        Z = np.dot(W,A)+b
        cache = {"A":A, "W":W, "b":b}
        return Z, cache

    def linear_activation_forward(self,layer):
        W,b = layer.weights["W"], layer.weights["b"]
        Z, linear_cache = self.linear_forward(layer.prev.A(), W, b)
        Z = self.apply_prefilter(layer, Z)
        A_, activation_cache = ActivationFunctions().forward[layer.activation.value](Z)
        A_ = self.apply_postfilter(layer, A_)
        cache = {"linear":linear_cache, "activation":activation_cache}
        return A_, cache

    def apply_prefilter(self, layer, Z):
        for f in layer.filters[LayerType.PreFilter.value]:
            Z = f.func(Z)
        return Z

    def apply_postfilter(self, layer, A):
        for f in layer.filters[LayerType.PostFilter.value]:
            A = f.func(A)
        return A

    def propagateForward(self, layer):
        Aprev,cache = self.linear_activation_forward(layer)
        layer.X = Aprev 
        layer.cache = cache

    def forward(self):
        for hlayer in self.nn.hidden_layers:
            self.propagateForward(hlayer)
        self.propagateForward(self.nn.output_layer)

    def propagateBackward(self):
        pass
        print (self.name, "propagate backward")


    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache["A"], cache["W"], cache["b"]
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, layer):
        dZ = ActivationFunctions().backward[layer.activation.value](layer.grads["dA"], layer.cache["activation"])
        return self.linear_backward(dZ, layer.cache["linear"]) 

    def backward(self, Y):
        AL = self.nn.output_layer.A()
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)


        self.nn.output_layer.grads["dA"] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        dA, dW, db = self.linear_activation_backward(self.nn.output_layer)
        self.nn.output_layer.grads["dW"] = dW
        self.nn.output_layer.grads["db"] = db
        self.nn.output_layer.prev.grads["dA"] = dA

        for hlayer in reversed(self.nn.hidden_layers):
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(hlayer)
            hlayer.grads["dW"] = dW_temp
            hlayer.grads["db"] = db_temp
            hlayer.prev.grads["dA"] = dA_prev_temp

    def compute_cost(self):
        AL = self.nn.output_layer.A()
        Y = self.nn.Y
        m = Y.shape[1]
        cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
        return np.squeeze(cost)

    def epoch(self,  n, learning_rate=0.001):
        for i in range(n):
            self.forward()
            self.nn.cost_layer.cost = self.compute_cost()
            self.nn.costs.append(self.nn.cost_layer.cost)
            self.backward(self.nn.Y)
            self.update_weights(learning_rate)

    def update_weights(self, learning_rate=0.001):
        for layer in self.nn.hidden_layers:
            layer.weights["W" ] = layer.weights["W"] - learning_rate * layer.grads["dW"]
            layer.weights["b"]  = layer.weights["b"] - learning_rate * layer.grads["db"]
        self.nn.output_layer.weights["W" ] = self.nn.output_layer.weights["W"] - learning_rate * self.nn.output_layer.grads["dW"]
        self.nn.output_layer.weights["b"]  = self.nn.output_layer.weights["b"] - learning_rate * self.nn.output_layer.grads["db"]


    def predict(self, X):
        self.forward()
        return  self.nn.output_layer.A().flatten()
        
    def optimize(self,num_iterations, learning_rate):
        self.epoch(num_iterations, learning_rate)







