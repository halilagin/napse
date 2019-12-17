# napse
A new deep learning framework


```python

nn =      Layer("input_layer",(2,1)) \
        > Layer("h1", (5,1)) \
        > Layer("output_layer",(1,1), activation=Activation.SIGMOID) \
        > CostLayer("cost_layer",(1,1))

nn.train(train_X,train_Y, num_iterations=15000, learning_rate=0.2)

```
