#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import sklearn as sk
from sklearn.metrics import f1_score
import keras as k
import keras.utils.np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Merge, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
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
  train_y = k.utils.np_utils.to_categorical(np.array(train_y), classes)  
  test_x = sequence.pad_sequences(test_x, maxlen=maxlen)
  test_y = k.utils.np_utils.to_categorical(np.array(test_y), classes)  

  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  
  # convolutional branch
  left = Sequential()
  left.add(Embedding(len(dataset.alphabet),
                     cfg.getint('cnn', 'embdims'),
                     input_length=maxlen,
                     weights=None)) 
  left.add(Convolution1D(nb_filter=cfg.getint('cnn', 'filters'),
                          filter_length=2,
                          border_mode='valid',
                          activation='relu',
                          subsample_length=1))
  left.add(MaxPooling1D(pool_length=2))
  left.add(Flatten())

  # lstm branch
  right = Sequential()
  right.add(Embedding(len(dataset.alphabet),
                      cfg.getint('lstm', 'embdims'),
                      input_length=maxlen,
                      dropout=0.2,
                      weights=None)) 
  right.add(LSTM(64))

  # now merge
  model = Sequential()
  model.add(Merge([left, right], mode='concat'))
  model.add(Dropout(cfg.getfloat('cnn', 'dropout')))
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit([train_x, train_x],
            train_y,
            nb_epoch=cfg.getint('cnn', 'epochs'),
            batch_size=cfg.getint('cnn', 'batches'),
            verbose=1,
            validation_split=0.1)

  # distribution over classes
  distribution = model.predict([test_x, test_x],
                               batch_size=cfg.getint('cnn', 'batches'))
  # class predictions
  predictions = np.argmax(distribution, axis=1)
  # gold labels
  gold = np.argmax(test_y, axis=1)
  # f1 for each class
  f1 = f1_score(gold, predictions, average=None)

  print 'f1 for contains:', f1[1]
  print 'all scores:', f1
