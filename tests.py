#!/usr/bin/env python3.7
from napse import * 


tests=[]
#precedence test


class GrammarTest():

    def test1(self):
        nn = Layer("L1", (1,1)) | PreFilter("F1") 
        assert (isinstance(nn, LayerClass))
        assert (len(nn.filters)==1)

    def test2(self):
        layer = Layer("L1", (1,1)) | PreFilter("F1") | PreFilter("F2") 
        assert (isinstance(layer, LayerClass))
        assert (len(layer.filters)==2)
        assert (layer.filters[0].name=="F1")
        assert (layer.filters[1].name=="F2")

    def test3(self):
        nn = Layer("L1", (1,1)) | PreFilter("F1") | PreFilter("F2") > CostLayer()
        assert (isinstance(nn, Napse))
        assert (nn!=None)
        assert (nn.input_layer!=None)
        assert (nn.input_layer.name=="L1")
        assert (nn.output_layer!=None)
        assert (nn.cost_layer!=None)
        assert (len(nn.input_layer.filters)==2)
        assert (nn.input_layer.filters["pre"][0].name=="F1")
        assert (nn.input_layer.filters["pre"][1].name=="F2")

    def test4(self):
        f = lambda x:x
        nn = InputLayer("input_layer", (1,1)) \
                    | PreFilter("F1", f) \
                    | PostFilter("F2", f) \
                > Layer("h1", shape=(4,1), activation=Activation.RELU) \
                    | PreFilter("F3", f) \
                    | PostFilter("F4", f) \
                > Layer("output_layer", shape=(4,1), activation=Activation.SIGMOID) \
                > CostLayer()
        assert (isinstance(nn, Napse))
        assert (nn!=None)
        assert (len(nn.hidden_layers)==1) # 1 hidden layer, 1 output layer
        assert (len(nn.input_layer.filters)==2)
        assert (len(nn.hidden_layers[0].filters)==2)
        assert (nn.input_layer.filters[LayerType.PreFilter.value][0].name=="F1")
        assert (nn.input_layer.filters[LayerType.PostFilter.value][0].name=="F2")
        assert (nn.hidden_layers[0].filters[LayerType.PreFilter.value][0].name=="F3")
        assert (nn.hidden_layers[0].filters[LayerType.PostFilter.value][0].name=="F4")



#def all():
#   method_list = [func for func in dir(Tests) if callable(getattr(Tests, func))]
#    return method_list


#GrammarTest().test1()
#GrammarTest().test2()
#GrammarTest().test3()
GrammarTest().test4()

