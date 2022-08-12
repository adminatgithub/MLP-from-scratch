import numpy as np
from functions import *

class MLP:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x, batch_size):
    out = np.array(x)
    for l in self.layers:
      out = l.forward(out, batch_size)
    return out

  def loss_gradient(self, y):
    return self.layers[-1].y - y

  def loss(self, y):
    return 0.5*(self.layers[-1].y - y)**2

  def back(self, y, lr, batch_size):
    loss = self.loss(y)
    dLdY = self.loss_gradient(y)

    for i in reversed(range(len(self.layers))):
      if self.layers[i].trainable:

        self.layers[i].bias -= np.average(dLdY, axis=0) * lr   
        self.layers[i].weight -= np.dot(self.layers[i].dY_dX(), dLdY) / batch_size * lr
        dLdY = np.dot(dLdY, self.layers[i].weight.T)

      else:
        if self.layers[i].flatten:
            dLdY = dLdY.reshape(self.layers[i].x.shape)
        else: 
          dLdY = dLdY * self.layers[i].dY_dX()
    return np.sum(loss)

  def validate(self, x_test, y_test):
    score = 0
    for i in range(len(x_test)):
      pred = decode(self.forward(x_test[i], 1))
      if pred == y_test[i]:
        score += 1
    return score/len(x_test)

  def fit(self, x_train, y_train, x_test, y_test, lr, epochs, batch_size):

    for e in range(epochs):
      # ideally the samples should be shuffled at this point
      loss = 0
      for i in range(0, len(x_train), batch_size):
        if len(x_train)-i >= batch_size:
          x_batch = x_train[i:i+batch_size]
          y_batch = y_train[i:i+batch_size]

        self.forward(x=x_batch, batch_size=batch_size)
        loss += self.back(y=batch_encode(y_batch, 10), lr=lr, batch_size=batch_size)

      test_accuracy = self.validate(x_test=x_test, y_test=y_test)
      training_accuracy = self.validate(x_test=x_train[:2000], y_test=y_train[:2000])

      if e%10 == 0 or e%10 == 5: #prints the current metrics every 5 iterations
        print(f"{e}/{epochs} EPOCHS, TEST ACCURACY: {test_accuracy}, TRAINING ACCURACY: {training_accuracy}, LOSS: {loss}")
