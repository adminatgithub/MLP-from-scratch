class DenseLayer:
  def __init__(self, inputs, outputs):
    self.x = None
    self.y = None
    self.inputs = inputs
    self.weight = np.random.normal(size=(inputs, outputs))/inputs
    self.bias = np.zeros((1, outputs))
    self.trainable = True
    self.flatten = False

  def forward(self, x, batch_size):
    self.x = x
    
    self.y = np.dot(x, self.weight) + self.bias
    return self.y

  def dY_dX(self):
    return self.x.T

class InputLayer:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False
    self.flatten = False

  def forward(self, x, batch_size):
    self.x = self.y = x
    return self.y

  def dY_dX(self):
    return self.x
  
class Flatten:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False
    self.flatten = True

  def forward(self, x, batch_size):
    self.x = x
    self.y = x.reshape(batch_size, -1)
    return self.y
