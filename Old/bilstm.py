#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import sklearn as sk
from sklearn.metrics import f1_score
import keras as k
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
import dataset
import ConfigParser

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read('settings.ini')
  
  # learn alphabet from training data
  dataset = dataset.DatasetProvider([cfg.get('data', 'train'),
                                     cfg.get('data', 'test')])
  # now load training examples and labels
  train_x, train_y = dataset.load(cfg.get('data', 'train'))
  # now load test examples and labels
  test_x, test_y = dataset.load(cfg.get('data', 'test'))

  # turn x and y into numpy array among other things
  maxlen = max([len(seq) for seq in train_x + test_x])
  classes = len(set(train_y))
  train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
  train_y = np_utils.to_categorical(np.array(train_y), classes)  
  test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
  test_y = np_utils.to_categorical(np.array(test_y), classes)  

  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape, '\n'
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  
  left = k.models.Sequential()
  left.add(Embedding(len(dataset.alphabet),
                      cfg.getint('lstm', 'embdims'),
                      input_length=maxlen,
                      dropout=0.2,
                      weights=None)) 
  left.add(LSTM(64, go_backwards=False))

  right = k.models.Sequential()
  right.add(Embedding(len(dataset.alphabet),
                      cfg.getint('lstm', 'embdims'),
                      input_length=maxlen,
                      dropout=0.2,
                      weights=None)) 
  right.add(LSTM(64, go_backwards=True))

  model = k.models.Sequential()
  model.add(Merge([left, right], mode='concat'))
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit([train_x, train_x],
            train_y,
            nb_epoch=cfg.getint('lstm', 'epochs'),
            batch_size=cfg.getint('lstm', 'batches'),
            verbose=1,
            validation_split=0.1)

  # distribution over classes
  distribution = model.predict([test_x, test_x],
                               batch_size=cfg.getint('lstm', 'batches'))
  # class predictions
  predictions = np.argmax(distribution, axis=1)
  # gold labels
  gold = np.argmax(test_y, axis=1)
  # f1 for each class
  f1 = f1_score(gold, predictions, average=None)

  print 'f1 for contains:', f1[1]
  print 'all scores:', f1
