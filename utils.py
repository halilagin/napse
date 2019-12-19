
from enum import Enum




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
    SGDParameter = "SGDParameter"
