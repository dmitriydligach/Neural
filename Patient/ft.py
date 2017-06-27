#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os, numpy
import keras as k
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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

MODEL_PATH = '../Codes/model.h5'
MAXLEN = 1387 # must be same as in source task

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))

  dataset = dataset.DatasetProvider(data_dir)
  x, y = dataset.load()
  x = pad_sequences(x, maxlen=MAXLEN)
  print 'x shape:', x.shape
  print 'y len:', len(y)

  # make vectors for target task
  model = load_model(MODEL_PATH)
  interm_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('ptvec').output)
  interm_output = interm_layer_model.predict(x)
  print 'x shape:', interm_output.shape

  # save vectors for debugging
  numpy.savetxt('data.txt', interm_output)

  # conduct svm training
  
