import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn import datasets

class ActivationFunctions():





    def __init__(self):
        pass
        
        self.functions = {
            "backward":
                {
                "relu":         lambda dA_, cache_: self.relu_backward(dA_, cache_),
                "sigmoid":      lambda dA_, cache_: self.sigmoid_backward(dA_, cache_)
                },
            "forward":
                {
                "relu":         lambda Z_: self.relu_forward(Z_),
                "sigmoid":      lambda Z_: self.sigmoid_forward(Z_)
                }
        }
    def __getattr__(self, attr):
        return self.functions[attr]

    def sigmoid_forward(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def relu_forward(self,Z):
        A = np.maximum(0,Z)  
        cache = Z 
        return A, cache

    def sigmoid_backward(self,dA, cache):
        Z = cache 
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True) # just converting dz to a correct object.
        dZ[Z <= 0] = 0
        return dZ


