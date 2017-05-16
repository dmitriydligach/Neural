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
import dataset

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))

  dataset = dataset.DatasetProvider(train_dir, code_file)
  x, y = dataset.load()
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
  maxlen = max([len(seq) for seq in train_x])

  # turn x into numpy array among other things
  classes = len(dataset.code2int)
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

  model = Sequential()
  model.add(Embedding(len(dataset.token2int),
                      cfg.getint('cnn', 'embdims'),
                      input_length=maxlen))
  model.add(GlobalAveragePooling1D())

  model.add(Dense(cfg.getint('cnn', 'hidden')))
  model.add(Activation('relu'))

  model.add(Dense(classes))
  model.add(Activation('sigmoid'))

  optimizer = RMSprop(lr=cfg.getfloat('cnn', 'learnrt'))
  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            epochs=cfg.getint('cnn', 'epochs'),
            batch_size=cfg.getint('cnn', 'batch'),
            validation_split=0.0)

  # probability for each class; (test size, num of classes)
  distribution = model.predict(test_x, batch_size=cfg.getint('cnn', 'batch'))

  # turn into an indicator matrix
  distribution[distribution < 0.5] = 0
  distribution[distribution >= 0.5] = 1

  f1 = f1_score(test_y, distribution, average='macro')
  print "f1 =", f1
