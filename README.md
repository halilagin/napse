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


testing multiple optimizers.

```python


nn =    Layer("input_layer",(2,1)) \
          | RandomInitializer()\
      > Layer("h1", (5,1)) \
          | PreFilter("Filter1", filter_dummy)\
          | PostFilter("Filter2", filter_dummy)\
      > Layer("output_layer",(1,1), activation=Activation.SIGMOID)\
      > CostLayer()\
          | Optimizer( Adam(epochs=100, lr=0.1, batch_size=64) )\
          | -Optimizer( MiniBatchGD(epochs=1500, lr=0.2, batch_size=64) )\
          | -Optimizer( SGD(epochs=15000, lr=0.1, batch_size=64) )\
          | -Optimizer( GD(epochs=15000, lr=0.2) )

```
