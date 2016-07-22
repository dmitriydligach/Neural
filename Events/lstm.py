#!/usr/bin/env python

"""
Notes
-----

Number of training examples is the number of words in training data.
Thus, input_length to LSTM is wc -w train.txt.
"""

import numpy as np
np.random.seed(1337)

import sys
sys.dont_write_bytecode = True

import sklearn as sk
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from dataset import DatasetProvider
import word2vec_model

train_path = '/Users/Dima/Loyola/Data/Thyme/Deep/Events/train.txt'
test_path = '/Users/Dima/Loyola/Data/Thyme/Deep/Events/dev.txt'
emb_path = '/Users/Dima/Loyola/Data/Word2VecModels/mimic.txt'

batch = 50
epochs = 1

if __name__ == "__main__":

  train = DatasetProvider(train_path)
  train_x, train_y = train.load(emb_path)
  train_y = to_categorical(train_y, 2)
  test = DatasetProvider(test_path)
  test_x, test_y = test.load(emb_path)
  test_y = to_categorical(test_y, 2)

  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  
  model = Sequential()
  model.add(LSTM(128, input_length=247922, input_dim=300))
  model.add(Dense(2))
  model.add(Activation('softmax'))
  
  # model.add(Dense(1))
  # model.add(Activation('sigmoid'))
  
  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=batch,
            verbose=1,
            validation_split=0.1)
  score, accuracy = model.evaluate(test_x,
                                   test_y,
                                   batch_size=batch,
                                   verbose=1)
  print 'fold %d accuracy: %f' % (fold_num, accuracy)
  scores.append(accuracy)
  
  print np.mean(scores)
