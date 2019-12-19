
import numpy as np
from utils import *
from napse import *


class Initializers():

    def __init__(self):
        pass

    @staticmethod
    def init_weights_random(nn):
        for layer in nn.hidden_layers:
            layer.weights["W"] = np.random.randn(layer.shape[0], layer.prev.shape[0]) * 0.01
            layer.weights["b"] = np.zeros((layer.shape[0], 1)) * 0.01
        nn.output_layer.weights["W"] = np.random.randn(nn.output_layer.shape[0], nn.output_layer.prev.shape[0]) * 0.01
        nn.output_layer.weights["b"] = np.zeros( (nn.output_layer.shape[0], 1) ) * 0.01


    @staticmethod
    def init_weights_he(nn):
        for layer in nn.hidden_layers:
            layer.weights["W"] = np.random.randn(layer.shape[0], layer.prev.shape[0]) * np.sqrt(2 / layer.prev.shape[0])
            layer.weights["b"] = np.zeros((layer.shape[0], 1)) 
        nn.output_layer.weights["W"] = np.random.randn(nn.output_layer.shape[0], nn.output_layer.prev.shape[0]) * np.sqrt(2 / nn.output_layer.prev.shape[0])
        nn.output_layer.weights["b"] = np.zeros( (nn.output_layer.shape[0], 1) )


def RandomInitializer():
    name="Random initializer"
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.WeightInitializer
    layer_.func = Initializers.init_weights_random
    return layer_;

def HeInitializer():
    name="HE initializer"
    layer_ = LayerClass(name=name, shape=None, activation=None)
    layer_.type=LayerType.WeightInitializer
    layer_.func = Initializers.init_weights_he
    return layer_;
