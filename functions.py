import numpy as np

def encode(index, size):
  vec = [0. for i in range(size)]
  vec[index] = 1.
  return vec

def decode(vec):
  index = None
  h = 0
  for i,v in enumerate(vec[0]):
    if h<v:
      h = v
      index = i
  return index

def batch_encode(indexes, size):
  vectors = []
  for i in range(len(indexes)):
    vec = np.zeros((size))
    vec[indexes[i]] = 1
    vectors = np.append(vectors, vec)
  return vectors.reshape(-1, size)

def batch_decode(vectors):
  indexes = []
  for vec in vectors:
    index = None
    h = 0
    for i,v in enumerate(vec):
      if h<v:
        h = v
        index = i
    indexes.append(index)
  return indexes
