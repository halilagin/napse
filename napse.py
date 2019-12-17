#!/usr/bin/env python3.7
#import cv2
import numpy as np
import scipy
import sklearn
from activation_functions import ActivationFunctions
from enum import Enum

'''
see: https://www.geeksforgeeks.org/operator-overloading-in-python/
'''


class Activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"

class NapseBase():
    def __init__(self, cost_layer):
        self.input_layer = None
        self.output_layer = None
        self.cost_layer = cost_layer
        self.hidden_layers = []

class LayerBase():
    def __init__(self, name,shape, activation=Activation.RELU):
        self.weights = {"W":None, "b":None}
        self.X=None
        self.grads = {"dA":None,"dW":None, "db":None}
        self.input_layer=self
        self.output_layer=self
        self.cost_layer=self
        self.prev=self
        self.next=self
        assert (name!=None), "layer name cannot be None"
        self.name=name
        self.shape=shape
        self.size=np.prod(shape)
        self.cache=None
        self.activation=activation


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
        A_, activation_cache = ActivationFunctions().forward[layer.activation.value](Z)
        cache = {"linear":linear_cache, "activation":activation_cache}
        return A_, cache

    def propagateForward(self, layer):
        Aprev,cache = self.linear_activation_forward(layer)
        layer.X = Aprev 
        layer.cache = cache

    def forward(self):
        for hlayer in self.nn.hidden_layers:
            self.propagateForward(hlayer)
        Aprev, cache = self.linear_activation_forward(self.nn.output_layer)
        self.nn.output_layer.X = Aprev
        self.nn.output_layer.cache = cache

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

class Napse(NapseBase):

    def __init__(self, cost_layer):
        super().__init__(cost_layer)
        self.tie_layers()
        self.costs=[]
        self.optimizer = NapseOptimizerGD(self)

    def tie_layers(self):
        pass
        self.output_layer = self.cost_layer.prev
        hidden_layer = self.output_layer.prev
        while hidden_layer!=hidden_layer.prev:
            self.hidden_layers.insert(0,hidden_layer)
            hidden_layer = hidden_layer.prev
        self.input_layer = hidden_layer


    def train(self,X, Y, num_iterations=1, learning_rate=0.001, weights=None):
        self.X = X
        self.Y = Y
        self.input_layer.X = X
        if weights==None:
            self.init_weights()
        else:
            self.set_weights(weights)
        self.optimizer.optimize(num_iterations, learning_rate)


    def predict(self, X):
        self.X = X
        self.input_layer.X = X
        m = X.shape[1]
        p = np.zeros((1,m))
        prediction_ = self.optimizer.predict(X)
        return prediction_

    def set_weights(self,weights):
        layer = self.input_layer.next #first hidden layer
        i=0
        while id(layer)!=id(layer.next):
            layer.weights = weights[i]
            layer = layer.next
            i+=1

    def init_weights(self):
        for layer in self.hidden_layers:
            layer.weights["W"] = np.random.randn(layer.shape[0], layer.prev.shape[0]) * 0.01
            layer.weights["b"] = np.zeros((layer.shape[0], 1)) * 0.01
        self.output_layer.weights["W"] = np.random.randn(self.output_layer.shape[0], self.output_layer.prev.shape[0]) * 0.01
        self.output_layer.weights["b"] = np.zeros( (self.output_layer.shape[0], 1) ) * 0.01
   
    def __str__(self):
        str_ = ""
        layer=self.input_layer
        while layer!=None:
            str_ = str_ + "%s\n" % (layer)
            if id(layer)==id(layer.next):
                break
            layer = layer.next
        return  str_.rstrip()



class Layer(LayerBase):

    def __init__(self, name,shape, activation=Activation.RELU):
        super().__init__(name, shape, activation)
        self.napse = None


    def A(self):
        return self.X

    def __gt__(self, next_layer):
        self.next = next_layer
        next_layer.prev = self
        if isinstance(next_layer, CostLayer):
            next_layer.napse = Napse(next_layer)
            return next_layer.napse
        return next_layer # to be able to propogate '>' to tbe next layer


    def __lt__(self, layer_):
        self.prev = layer_

    def __add__(self, other):
        pass
        return self

    def __sub__(self, other):
        pass
        return self

    def __pow__(self, other):
        pass
        return self

    def compile(self ):
        return self.input_layer

    def __str__(self):
        return "%s, %d" % (self.name, self.size)

class CostLayer(LayerBase):
    pass

    def __init__(self, name,shape):
        super().__init__(name, shape)
        self.cost=None




#def __init__(self, name, size)
#    super().__init__(name, size)

