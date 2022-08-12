from keras.datasets import mnist
from mlp import MLP
from layers import *
from activations import *


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.*2.-1.
x_test = x_test/255.*2.-1.


layers = [
  InputLayer(),
  Flatten(),
  DenseLayer(784, 60),
  Relu(),
  DenseLayer(60, 10),
  Sigmoid(),
]

mlp = MLP(layers)


mlp.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,lr=0.01, epochs=10, batch_size=128))
