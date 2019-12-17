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


class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"

class NapseBase():
    def __init__(self, cost_layer):
        self.input_layer = None
        self.output_layer = None
        self.cost_layer = cost_layer
        self.hidden_layers = []

class LayerBase():
    def __init__(self, name,shape):
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

class NapseGradientDescent():
    pass

    def __init__(self, napse):
        pass


class Napse(NapseBase):

    def __init__(self, cost_layer):
        super().__init__(cost_layer)
        self.tie_layers()
        self.costs=[]
        self.optimizer = NapseOptimezer(self)

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
        self.epoch(num_iterations, learning_rate)
        
        probas = self.output_layer.A()



    def predict(self, X):
        self.X = X
        self.input_layer.X = X
        m = X.shape[1]

        p = np.zeros((1,m))
        
        self.input_layer.forward()
        probas = self.output_layer.A().flatten()
        probas[probas>=0.5]=1
        probas[probas<0.5]=0
        return probas

    def set_weights(self,weights):
        layer = self.input_layer.next #first hidden layer
        i=0
        while id(layer)!=id(layer.next):
            layer.weights = weights[i]
            layer = layer.next
            i+=1

    def init_weights(self):
        layer = self.input_layer.next #first hidden layer
        while id(layer)!=id(layer.next):
            layer.weights["W"] = np.random.randn(layer.shape[0], layer.prev.shape[0]) * 0.01
            layer.weights["b"] = np.random.randn(layer.shape[0], 1) * 0.01
            layer = layer.next

    def compute_cost(self ):
        AL = self.output_layer.A()
        Y = self.Y
        m = Y.shape[1]
        cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
        return np.squeeze(cost)

    def epoch(self,  n, learning_rate=0.001):
        for i in range(n):
            self.input_layer.forward()
            self.cost_layer.cost = self.compute_cost()
            self.costs.append(self.cost_layer.cost)
            self.output_layer.backward(self.Y)
            self.update_weights(learning_rate)

    def update_weights(self, learning_rate=0.001):
        # Update rule for each parameter
        layer = self.input_layer.next
        while isinstance(layer,CostLayer)==False:
            layer.weights["W" ] = layer.weights["W"] - learning_rate * layer.grads["dW"]
            layer.weights["b"]  = layer.weights["b"] - learning_rate * layer.grads["db"]
            layer = layer.next


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

    def __init__(self, name,shape):
        super().__init__(name, shape)
        self.napse = None

    def A(self):
        return self.X



    def linear_forward(self,A, W, b):
        Z = np.dot(W,A)+b
        cache = {"A":A, "W":W, "b":b}
        return Z, cache

    def linear_activation_forward(self,  activation):
        W = self.weights["W"]
        b = self.weights["b"]
        Z, linear_cache = self.linear_forward(self.prev.A(), W, b)
        A_, activation_cache = ActivationFunctions().forward[activation.value](Z)
        #A_, activation_cache = self.relu_forward(Z)
        cache = {"linear":linear_cache, "activation":activation_cache}
        return A_, cache

    def propagateForward(self):
        pass
        Aprev,cache = self.linear_activation_forward(activation = ActivationType.RELU)
        self.X = Aprev 
        self.cache = cache



    def propagateBackward(self):
        pass
        print (self.name, "propagate backward")




    def forward(self):
        #go forward in hidden layers
        layer=self.next #first hidden layer
        while True:
            if isinstance (layer.next, CostLayer)==True: # stop when it is layer.next=output_layer
                break
            layer.propagateForward()
            if id(layer)==id(layer.next):
                break
            layer = layer.next

        #go forward for the outputlayer
        output_layer = layer
        Aprev, cache = output_layer.linear_activation_forward(activation = ActivationType.SIGMOID)
        output_layer.X = Aprev
        output_layer.cache = cache


    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache["A"], cache["W"], cache["b"]
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache["linear"], cache["activation"]
        dZ = ActivationFunctions().backward[activation.value](dA, activation_cache)
        return self.linear_backward(dZ, linear_cache) 

    def backward(self, Y):
        #starts from output_layer=self
        output_layer = self
        AL = output_layer.A()
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)


        output_layer.grads["dA"] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        dA, dW, db = self.linear_activation_backward(output_layer.grads["dA"], output_layer.cache, ActivationType.SIGMOID)
        output_layer.grads["dW"] = dW
        output_layer.grads["db"] = db
        output_layer.prev.grads["dA"] = dA



        hlayer=output_layer.prev #hlayer -> hidden layer
        while hlayer!=hlayer.prev:
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                    hlayer.grads["dA"], 
                    hlayer.cache, 
                    ActivationType.RELU)
            hlayer.grads["dW"] = dW_temp
            hlayer.grads["db"] = db_temp
            hlayer.prev.grads["dA"] = dA_prev_temp
            hlayer = hlayer.prev


    #def __gt__(self, layer_):
    #    self.next = layer_
    #    layer_.prev = self
    #    layer_.inherite_from_prev_layer(self)
    #    self.delegate_to_next_layer(layer_)
    #    return layer_ # to be able to propogate '>' to tbe next layer
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

