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
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from dataset_for_mlp import DatasetProvider
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

  # model.add(LSTM(128, input_dim=300, input_length=train_x.shape[0]))
  # model.add(LSTM(128))

  model.add(Dense(128, input_dim=300))
  model.add(Activation('relu'))

  model.add(Dense(2))
  model.add(Activation('softmax'))
  
  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=batch,
            verbose=1,
            validation_split=0.1)
  
  # probability for each class; (test size, num of classes)
  distribution = model.predict(test_x, batch_size=batch)
  # class predictions; (test size,)
  predictions = np.argmax(distribution, axis=1)
  # gold labels; (test size,)
  gold = np.argmax(test_y, axis=1)

  # f1 scores
  label_f1 = f1_score(gold, predictions, average=None)
  positive_class_index = 1
  print 'f1:', label_f1[positive_class_index]
  
