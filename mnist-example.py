import numpy as np
import random
from keras.datasets import mnist
import timeit


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.*2.-1.
x_test = x_test/255.*2.-1.
