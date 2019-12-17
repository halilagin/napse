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
from optimizers import *
from utils import *

'''
see: https://www.geeksforgeeks.org/operator-overloading-in-python/
'''

class NapseBase():
    def __init__(self, cost_layer):
        self.input_layer = None
        self.output_layer = None
        self.cost_layer = cost_layer
        self.hidden_layers = []


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





class Activation(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"


'''
precedence: top precedes bottom
**  Exponentiation
~x  Bitwise not
+x, -x  Positive, negative
*, /, % Multiplication, division, remainder
+, -    Addition, subtraction
<<, >>  Bitwise shifts
&   Bitwise AND
^   Bitwise XOR
|   Bitwise OR
in, not in, is, is not, <, <=,  >,  >=,
<>, !=, ==

'''

class LayerBase():
    def __init__(self, name,shape, activation=Activation.RELU):
        self.type=None  # "input_layer", "hidden_layer" or "cost_layer"
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



#see: https://www.journaldev.com/26737/python-bitwise-operators
class LayerClass(LayerBase):

    def __init__(self, name,shape, activation=Activation.RELU):
        super().__init__(name, shape, activation)
        self.filters={LayerType.PreFilter.value:[],LayerType.PostFilter.value:[]}
        self.napse = None

        #this is pointer to a function that can be applied on a layer's data, X or sigma(X), grads.
        #this is valid when the layer.type=filter
        self.func = None 


    def A(self):
        return self.X


    def __or__(self, first_filter ):
        first_filter.layer = self
        self.filters[first_filter.type.value].append(first_filter)
        return self

    def __gt__(self, next_layer):
        self.next = next_layer
        next_layer.prev = self
        if next_layer.type==LayerType.CostLayer:
            next_layer.napse = Napse(next_layer)
            return next_layer.napse
        return next_layer # to be able to propogate '>' to tbe next layer

    def gt(self, next_layer):
        self.__gt__(next_layer)

        

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

class CostLayerClass(LayerClass):
    pass

    def __init__(self, name,shape):
        super().__init__(name, shape)
        self.cost=None


def InputLayer(name="input_layer", shape=(1,1), activation=Activation.RELU):
    layer_ = LayerClass(name, shape, activation)
    layer_.type=LayerType.InputLayer
    return layer_;

def HiddenLayer(name="hidden_layer", shape=(1,1), activation=Activation.RELU):
    layer_ = LayerClass(name, shape, activation)
    layer_.type=LayerType.HiddenLayer
    return layer_;

def Layer(name="hidden_layer", shape=(1,1), activation=Activation.RELU):
    return  HiddenLayer(name, shape, activation);

def OutputLayer(name="output_layer", shape=(1,1), activation=Activation.RELU):
    layer_ = LayerClass(name, shape, activation)
    layer_.type=LayerType.OutputLayer
    return layer_;

def Filter(name, func):
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.PreFilter
    layer_.func = func
    return layer_;

#operates on X in the layer
def PreFilter(name, func):
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.PreFilter
    layer_.func = func
    return layer_;

def Conv2D(name, func):
    return PreFilter(name, func)


#operates on sigma(X) in the layer
def PostFilter(name, func):
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.PostFilter
    layer_.func = func
    return layer_;

def CostLayer():
    layer_ = LayerClass(name="cost_layer", shape=None, activation=None)
    layer_.type=LayerType.CostLayer
    return layer_;

