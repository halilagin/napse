# napse
A new deep learning framework


```python


nn = InputLayer("input_layer", (1,1)) \
       | PreFilter("F1", f) \
       | PostFilter("F2", f) \
   > Layer("h1", shape=(4,1), activation=Activation.RELU) \
       | PreFilter("F3", f) \
       | PostFilter("F4", f) \
   > Layer("output_layer", shape=(4,1), activation=Activation.SIGMOID) \
   > CostLayer()

nn.train(train_X,train_Y, num_iterations=15000, learning_rate=0.2)

```
