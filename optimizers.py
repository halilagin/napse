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
        self.napse = napse


    def linear_forward(self,A,W,b):
        Z = np.dot(W,A)+b
        cache = {"A":A, "W":W, "b":b}
        return Z, cache

    def linear_activation_forward(self,layer):
        W,b = layer.weights["W"], layer.weights["b"]
        Z, linear_cache = self.linear_forward(layer.prev.A(), W, b)
        Z = self.apply_prefilter(layer, Z)
        A_, activation_cache = ActivationFunctions().forward[layer.activation.value](Z)

        #dropoupt filter is a postfilter and conflicts with the declared custom postfilters
        #make it exclusive or. keep d1 dropout matrix for backpropagation
        #dropout is not applied on output_layer
        dropouts=None 
        if self.napse.filter_exists(LayerType.DropOutRegularizer.value):
            if layer!=self.napse.output_layer: 
                dropout_func = self.napse.cost_layer.filters[LayerType.DropOutRegularizer.value][0].properties["forward_func"]
                keep_prob = self.napse.cost_layer.filters[LayerType.DropOutRegularizer.value][0].properties["keep"]
                A_,dropouts = dropout_func(A_,keep_prob)
        else:
            A_ = self.apply_postfilter(layer, A_)

        cache = {"linear":linear_cache, "activation":activation_cache, "dropouts":dropouts}
        return A_, cache

    def apply_prefilter(self, layer, Z):
        if LayerType.PreFilter.value not in layer.filters.keys():
            return Z
        for f in layer.filters[LayerType.PreFilter.value]:
            Z = f.func(Z)
        return Z

    def apply_postfilter(self, layer, A):
        if LayerType.PostFilter.value not in layer.filters.keys():
            return A
        for f in layer.filters[LayerType.PostFilter.value]:
            A = f.func(A)
        return A

    def propagateForward(self, layer):
        Aprev,cache = self.linear_activation_forward(layer)
        layer.X = Aprev 
        layer.cache = cache

    def forward(self):
        for hlayer in self.napse.hidden_layers:
            self.propagateForward(hlayer)
        self.propagateForward(self.napse.output_layer)

    def propagateBackward(self):
        pass
        print (self.name, "propagate backward")


    def linear_backward(self, dZ, layer):
        A_prev, W, b = layer.cache["linear"]["A"], layer.cache["linear"]["W"], layer.cache["linear"]["b"]
        m = A_prev.shape[1]

        def default_regularizer(w_):
            return np.zeros(w_.shape)
        
        def l2_regularizer(lambda_, w):
            return lambda_ * w

        regularizer = default_regularizer

        #dropout filter is a regularizer and conflicts with the declared regularizers
        #make it exclusive or. read dropouts from cache and apply on dw. see backward function
        if self.napse.filter_exists(LayerType.DropOutRegularizer.value):
            pass
        else:
            if layer.napse.filter_exists(LayerType.L2Regularizer.value): 
                lambda_ = layer.napse.cost_layer.filters[LayerType.L2Regularizer.value][0].properties["lambda_"]
                regularizer = lambda w: l2_regularizer(lambda_, w)

        dW = 1./m * np.dot(dZ, A_prev.T) + regularizer(layer.weights["W"])
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, layer):
        dZ = ActivationFunctions().backward[layer.activation.value](layer.grads["dA"], layer.cache["activation"])
        return self.linear_backward(dZ, layer) 

    def backward(self, Y):

        def apply_dropout_if_it_exists(dA_, layer_):
            #dropout filter should not be applied on output_layer
            if layer_==self.napse.input_layer:
                return dA_
            if layer_ == self.napse.output_layer:
                return dA_
            if  self.napse.filter_exists(LayerType.DropOutRegularizer.value):
                dropout_backward_func = self.napse.cost_layer.filters[LayerType.DropOutRegularizer.value][0].properties["backward_func"]
                keep_prob = self.napse.cost_layer.filters[LayerType.DropOutRegularizer.value][0].properties["keep"]
                return dropout_backward_func(dA_,keep_prob, layer_.cache["dropouts"])
            return dA_


        AL = self.napse.output_layer.A()
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)


        self.napse.output_layer.grads["dA"] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        l_=self.napse.output_layer
        #loop on hidden layers and output layer but stop when it sees the first hidden layer
        while l_!=self.napse.input_layer:
            dA, dW, db = self.linear_activation_backward(l_)
            l_.grads["dW"] = dW
            l_.grads["db"] = db
            l_.prev.grads["dA"] = apply_dropout_if_it_exists(dA,l_.prev)#this also skips output_layer
            l_=l_.prev
            #dA = apply_dropout_if_it_exists(dA,l_)

    

    def compute_cost(self):
        AL = self.napse.output_layer.A()
        Y = self.napse.Y
        m = Y.shape[1]
        cost = (-1./m) * np.nansum( np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)) )
        if self.napse.filter_exists(LayerType.L2Regularizer.value): 
            lambda_ = self.napse.cost_layer.filters[LayerType.L2Regularizer.value][0].properties["lambda_"]
            cost += 1/m * lambda_ / 2 * self.l2_regularizer_cost()
        return cost

    def l2_regularizer_cost(self):
        w_sum = 0
        for layer_ in self.napse.hidden_layers:
            w_sum += np.nansum ( layer_.weights["W"]**2)
        w_sum += np.sum ( self.napse.output_layer.weights["W"]**2)
        return np.squeeze(w_sum)




    def epoch(self,  n, learning_rate=0.001):
        for i in range(n):
            self.forward()
            self.napse.cost_layer.cost = self.compute_cost()
            self.napse.costs.append(self.napse.cost_layer.cost)
            self.backward(self.napse.Y)
            self.update_weights(learning_rate)

    def update_weights(self, learning_rate=0.001):
        for layer in self.napse.hidden_layers:
            layer.weights["W" ] = layer.weights["W"] - learning_rate * layer.grads["dW"]
            layer.weights["b"]  = layer.weights["b"] - learning_rate * layer.grads["db"]
        self.napse.output_layer.weights["W" ] = self.napse.output_layer.weights["W"] - learning_rate * self.napse.output_layer.grads["dW"]
        self.napse.output_layer.weights["b"]  = self.napse.output_layer.weights["b"] - learning_rate * self.napse.output_layer.grads["db"]


    def predict(self, X):
        self.forward()
        return  self.napse.output_layer.A().flatten()
        
    def optimize(self,num_iterations, learning_rate):
        self.epoch(num_iterations, learning_rate)







