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
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
import dataset

def print_config(cfg):
  """Print configuration settings"""

  print 'train:', cfg.get('data', 'path')
  print 'batch:', cfg.get('nn', 'batch')
  print 'epochs:', cfg.get('nn', 'epochs')
  print 'embdims:', cfg.get('nn', 'embdims')
  print 'hidden:', cfg.get('nn', 'hidden')
  print 'learnrt:', cfg.get('nn', 'learnrt')

def get_model(cfg, num_of_features):
  """Model definition"""

  model = Sequential()
  model.add(Embedding(input_dim=num_of_features,
                      output_dim=cfg.getint('nn', 'embdims'),
                      input_length=maxlen))
  model.add(GlobalAveragePooling1D())

  model.add(Dropout(cfg.getfloat('nn', 'dropout')))
  model.add(Dense(cfg.getint('nn', 'hidden')))
  model.add(Activation('relu'))

  model.add(Dense(classes))
  model.add(Activation('softmax'))

  return model

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  print_config(cfg)

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))
  dataset = dataset.DatasetProvider(
    data_dir,
    cfg.getint('args', 'min_token_freq'))
  x, y = dataset.load()




  train_x, test_x, train_y, test_y = train_test_split(
    x,
    y,
    test_size=0.20)
  maxlen = max([len(seq) for seq in train_x])

  # turn x into numpy array among other things
  classes = len(dataset.label2int)
  train_x = pad_sequences(train_x, maxlen=maxlen)
  test_x = pad_sequences(test_x, maxlen=maxlen)
  train_y = to_categorical(train_y, classes)
  test_y = to_categorical(test_y, classes)
  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  print 'number of features:', len(dataset.token2int)

  model = get_model(cfg, len(dataset.token2int))

  optimizer = RMSprop(lr=cfg.getfloat('nn', 'learnrt'))
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            epochs=cfg.getint('nn', 'epochs'),
            batch_size=cfg.getint('nn', 'batch'),
            validation_split=0.0)

  # probability for each class; (test size, num of classes)
  distribution = model.predict(
    test_x,
    batch_size=cfg.getint('nn', 'batch'))
  # class predictions; (test size,)
  predictions = np.argmax(distribution, axis=1)
  # gold labels; (test size,)
  gold = np.argmax(test_y, axis=1)

  # f1 scores
  label_f1 = f1_score(gold, predictions, average=None)
  print label_f1
