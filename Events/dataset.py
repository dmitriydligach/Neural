#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python -B

"""
Some statistics:

tokens in data: 7463
embeddings: 131045
have embeddings for: 6237
"""

import numpy
import word2vec_model

GOLDPATH = '/Users/Dima/Loyola/Data/Thyme/events.txt'
EMBPATH = '/Users/Dima/Loyola/Data/Word2Vec/Models/mimic.txt'

class DatasetProvider:
  """Provide data"""

  def __init__(self, path=GOLDPATH):
    """Each token is a sample. Events are marked as [event]"""

    self.data = []
    self.labels = []
    
    for line in open(GOLDPATH):
      for token in line.split():
        if token.startswith('[') and token.endswith(']'):
          self.labels.append(1)   # this is an event
          self.data.append(token[1:-1])
        else:
          self.labels.append(0)   # this is not an event
          self.data.append(token)


  def load(self, path=EMBPATH):
    """Each token is a sample"""

    word2vec = word2vec_model.Model(path)
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

  dataset = DatasetProvider()
  dataset.load()
  
