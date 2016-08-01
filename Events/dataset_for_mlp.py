#!/usr/bin/env python

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import numpy
import word2vec_model

class DatasetProvider:
  """Provide data"""

  def __init__(self, data_path):
    """Each token is a sample. Events are marked as [event]"""

    self.data = []
    self.labels = []
    
    for line in open(data_path):
      for token in line.split():
        if token.startswith('[') and token.endswith(']'):
          self.data.append(token[1:-1])
          self.labels.append(1)   # this is an event
        else:
          self.data.append(token)
          self.labels.append(0)   # this is not an event

  def load(self, embed_path):
    """Each token is a sample"""

    word2vec = word2vec_model.Model(embed_path)
    uniq_words_in_data = set(self.data)
    average = word2vec.average_words(uniq_words_in_data)

    data = [] # list of numpy arrays
    for token in self.data:
      if token in word2vec.vectors:
        data.append(word2vec.vectors[token])
      else:
        data.append(average)

    return numpy.array(data), numpy.array(self.labels)

if __name__ == "__main__":

  train_path = '/Users/Dima/Loyola/Data/Thyme/Deep/Events/train.txt'
  test_path = '/Users/Dima/Loyola/Data/Thyme/Deep/Events/dev.txt'
  emb_path = '/Users/Dima/Loyola/Data/Word2VecModels/mimic.txt'
  
  dataset = DatasetProvider(train_path)
  dataset.load(emb_path)
  
