
from enum import Enum




class LayerType(Enum):
    InputLayer = "InputLayer"
    CostLayer = "CostLayer"
    HiddenLayer = "HiddenLayer"
    OutputLayer = "OutputLayer"
    PreFilter = "PreFilter"
    PostFilter = "PostFilter"
