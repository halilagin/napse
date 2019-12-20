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
        self.X_original = None
        self.Y_original = None
        self.Y=None
        self.batch_indexes=None
        self.optimization_algorithm=OptimizationAlgorithm
        self.optimizer = NapseOptimizerGD(self)

    def tie_layers(self):
        pass
        self.output_layer = self.cost_layer.prev
        hidden_layer = self.output_layer.prev
        while hidden_layer!=hidden_layer.prev:
            self.hidden_layers.insert(0,hidden_layer)
            hidden_layer = hidden_layer.prev
        self.input_layer = hidden_layer

# layer.napse.cost_layer.filters[LayerType.L2Regularizer.value][0].type
    def filter_exists(self, filter_type):
        l_ = self.input_layer
        while True:
            if l_.filter_exists(filter_type):
                return True
            if l_.type==LayerType.CostLayer:
                break
            l_ = l_.next
        return False

    def train(self,X, Y, epochs=1, learning_rate=0.001, weights=None):
        self.X_original = X
        self.Y_original = Y
        self.set_default_batch_indexes() # one single batch
        self.set_optimization_algorithm()
        if weights!=None:
            self.set_weights(weights)
        self.run_initializer() # if there is an initializier overwrite weights
        self.optimizer.optimize(epochs, learning_rate)


    def set_optimization_algorithm(self):
        if False == self.filter_exists(LayerType.Optimizer.value):
            self.optimization_algorithm = GD()
        else:
            self.optimization_algorithm = self.cost_layer.filters[LayerType.Optimizer.value][0] #there is always one in index=0

        if self.optimization_algorithm.properties["optimizer"].type == OptimizationAlgorithm.GD.value:
            self.set_default_batch_indexes()
        elif self.optimization_algorithm.properties["optimizer"].type == OptimizationAlgorithm.MiniBatchGD.value:
            batch_size = self.cost_layer.filters[LayerType.Optimizer.value][0].properties["optimizer"].batch_size
            self.batch_indexes = self.prepare_batch_indexes(self.X_original.shape[1], batch_size)
        elif self.optimization_algorithm.properties["optimizer"].type == OptimizationAlgorithm.SGD.value:
            batch_size = self.cost_layer.filters[LayerType.Optimizer.value][0].properties["optimizer"].batch_size
            self.batch_indexes = self.prepare_batch_indexes(self.X_original.shape[1], batch_size)
            


    def set_default_batch_indexes(self):
        # the default batch indexes is the lower and upper bound of X, one tuple of indexes
        self.batch_indexes = self.prepare_batch_indexes(self.X_original.shape[1], self.X_original.shape[1])

    def prepare_batch_indexes(self, arr_length, batch_size):
        indexes=[]
        for batch_lower in range(0,arr_length, batch_size):
            batch_upper = batch_lower+batch_size
            if batch_upper > arr_length:
                batch_upper = arr_length
            indexes.append([batch_lower,batch_upper])
        return indexes


    def run_initializer(self):
        pass
        initializer_ = None
        if len(self.input_layer.filters[LayerType.WeightInitializer.value])>0:
            initializer_ = self.input_layer.filters[LayerType.WeightInitializer.value][0]
        if initializer_!=None: #there is a weight initializer
            initializer_.func(self)
            

    def predict(self, X):
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
        self.filters= {} #{LayerType.PreFilter.value:[],LayerType.PostFilter.value:[], LayerType.WeightInitializer.value:[]}
        self.napse = None
        self.epoch_count=0 # how many epoches processed on this layer
        self.properties={} # filter properties

        #this is pointer to a function that can be applied on a layer's data, X or sigma(X), grads.
        #this is valid when the layer.type=filter
        self.func = None 

    def add_filter(self, filter_):
        if filter_.type.value not in self.filters:
            self.filters[filter_.type.value]=[]

        if filter_.type==LayerType.Optimizer:
            self.filters[filter_.type.value]=[filter_]# there can only be one optimizer
        else: 
            self.filters[filter_.type.value].append(filter_)

    def filter_exists(self, filter_type):
        return filter_type in self.filters

    def A(self):
        return self.X

    def __or__(self, first_filter ):
        first_filter.layer = self
        #self.filters[first_filter.type.value].append(first_filter)
        self.add_filter(first_filter)
        return self

    def __gt__(self, next_layer):

        def set_napse(layer, napse):
            #this layer is the cost_layer and build connection between napse and this layer.
            layer.napse = napse
            #besides, all layers should points to napse object too.
            l_ = layer.prev
            while l_!=l_.prev:
                l_.napse = napse
                l_=l_.prev

        self.next = next_layer
        next_layer.prev = self
        if next_layer.type==LayerType.CostLayer:
            set_napse(next_layer, Napse(next_layer))
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

def CostLayer():
    layer_ = LayerClass(name="cost_layer", shape=None, activation=None)
    layer_.type=LayerType.CostLayer
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


class GD():
    def __init__(self, batch_size=None):
        self.type=OptimizationAlgorithm.GD.value

class MiniBatchGD():
    def __init__(self, batch_size=None):
        self.type=OptimizationAlgorithm.MiniBatchGD.value
        self.batch_size=batch_size

class SGD():
    def __init__(self, batch_size=None):
        #layer_.func = func
        self.type=OptimizationAlgorithm.SGD.value
        self.batch_size=batch_size

def Optimizer(optimizer_implementation):
    name="Optimizer"
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.Optimizer
    layer_.properties["optimizer"]=optimizer_implementation
    return layer_;


def L2Regularizer(lambda_=0.1):
    name="L2 regularizer"
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.L2Regularizer
    #layer_.func = func
    layer_.properties["lambda_"]=lambda_
    return layer_;

def DropOutRegularizer(keep=0.8):
    name="Dropout regularizer"
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.DropOutRegularizer
    
    def forward_func(A, keep ):
        D1 = np.random.rand(*A.shape)
        D1 = D1 < keep       
        A = A * D1                
        A = A / keep  
        assert(np.isnan(A).any()==False)
        return A, D1
    def backward_func(dA, keep_prob, cache_dropouts):
        dA = dA * cache_dropouts
        dA = dA / keep_prob    
        assert(np.isnan(dA).any()==False)
        return dA

    layer_.properties["keep"]=keep
    layer_.properties["forward_func"]=forward_func
    layer_.properties["backward_func"]=backward_func
    return layer_;
