#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.models import Model
import dataset

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'corpus'))

  dataset = dataset.DatasetProvider(data_dir)
  x, y = dataset.load()
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
  maxlen = 1387

  # turn x into numpy array among other things
  classes = len(dataset.label2int)
  train_x = pad_sequences(train_x, maxlen=maxlen)
  test_x = pad_sequences(test_x, maxlen=maxlen)
  train_y = np.array(train_y)
  test_y = np.array(test_y)
  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  print 'unique features:', len(dataset.token2int)
  print 'train_x size in bytes:', train_x.size * train_x.itemsize

  model = load_model('../Codes/model.h5')
  intermediate_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer('ptvec').output)
  intermediate_output = intermediate_layer_model.predict(test_x)
  print intermediate_output.shape
