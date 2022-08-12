import numpy as np

class Sigmoid:
  def __init__(self):
      self.x = None
      self.y = None
      self.trainable = False
      self.flatten = False

  def forward(self, x, batch_size):
    self.x = x
    self.y = 1 / (1 + np.exp(-x))
    return self.y

  def dY_dX(self):
    return (1-self.y)*self.y
