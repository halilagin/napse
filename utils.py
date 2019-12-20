
from enum import Enum


class OptimizationAlgorithm(Enum):
    GD = "GD"
    SGD = "SGD"
    MiniBatchGD = "MiniBatchGD"
    ADAM = "ADAM"



class LayerType(Enum):
    InputLayer = "InputLayer"
    CostLayer = "CostLayer"
    HiddenLayer = "HiddenLayer"
    OutputLayer = "OutputLayer"
    PreFilter = "PreFilter"
    PostFilter = "PostFilter"
    WeightInitializer = "WeightInitializer"
    L2Regularizer = "L2Regularizer"
    DropOutRegularizer = "DropOutRegularizer"
    Optimizer = "Optimizer"
