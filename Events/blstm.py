#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.dont_write_bytecode = True

import sklearn as sk
from sklearn.metrics import f1_score
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Merge
import dataset
import ConfigParser

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  print 'train:', cfg.get('data', 'train')
  print 'test:', cfg.get('data', 'test')
  print 'batch:', cfg.get('lstm', 'batch')
  print 'epochs:', cfg.get('lstm', 'epochs')
  print 'embdims:', cfg.get('lstm', 'embdims')
  print 'units:', cfg.get('lstm', 'units')
  print 'dropout:', cfg.get('lstm', 'dropout')
  print 'udropout:', cfg.get('lstm', 'udropout')
  print 'wdropout:', cfg.get('lstm', 'wdropout')
  print 'learnrt:', cfg.get('lstm', 'learnrt')
  
  # learn alphabet from training data
  dataset = \
    dataset.DatasetProvider([cfg.get('data', 'train'),
                             cfg.get('data', 'test')])
  # now load training examples and labels
  train_x, train_y = dataset.load(cfg.get('data', 'train'))
  # now load test examples and labels
  test_x, test_y = dataset.load(cfg.get('data', 'test'))

  # turn x and y into numpy array among other things
  maxlen = max([len(seq) for seq in train_x + test_x])
  train_x = pad_sequences(train_x, maxlen=maxlen)
  train_y = pad_sequences(train_y, maxlen=maxlen)
  test_x = pad_sequences(test_x, maxlen=maxlen)
  test_y = pad_sequences(test_y, maxlen=maxlen)

  train_y =  np.array([to_categorical(seq, 3) for seq in train_y])
  test_y =  np.array([to_categorical(seq, 3) for seq in test_y])
  
  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape
  
  left = Sequential()
  left.add(Embedding(input_dim=len(dataset.word2int),
                      output_dim=cfg.getint('lstm', 'embdims'),
                      input_length=maxlen,
                      dropout=cfg.getfloat('lstm', 'dropout')))
  left.add(LSTM(cfg.getint('lstm', 'units'),
                 return_sequences=True,
                 go_backwards=False,
                 dropout_W = cfg.getfloat('lstm', 'wdropout'),
                 dropout_U = cfg.getfloat('lstm', 'udropout')))

  right = Sequential()
  right.add(Embedding(input_dim=len(dataset.word2int),
                      output_dim=cfg.getint('lstm', 'embdims'),
                      input_length=maxlen,
                      dropout=cfg.getfloat('lstm', 'dropout')))
  right.add(LSTM(cfg.getint('lstm', 'units'),
                 return_sequences=True,
                 go_backwards=True,
                 dropout_W = cfg.getfloat('lstm', 'wdropout'),
                 dropout_U = cfg.getfloat('lstm', 'udropout')))
  
  model = Sequential()
  model.add(Merge([left, right], mode='concat'))
  model.add(TimeDistributed(Dense(3)))
  model.add(Activation('softmax'))

  optimizer = RMSprop(lr=cfg.getfloat('lstm', 'learnrt'),
                      rho=0.9, epsilon=1e-08)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit([train_x, train_x],
            train_y,
            nb_epoch=cfg.getint('lstm', 'epochs'),
            batch_size=cfg.getint('lstm', 'batch'),
            verbose=1,
            validation_split=0.1)

  # distribution over classes (8645, 187, 3)
  distribution = \
    model.predict([test_x, text_x],
                  batch_size=cfg.getint('lstm', 'batch'))
  # class predictions (8645, 187)
  predictions = np.argmax(distribution, axis=2)
  # gold labels (8645, 187)
  gold = np.argmax(test_y, axis=2)

  # reshape into 1-d arrays
  total_labels = gold.shape[0] * gold.shape[1]
  predictions = predictions.reshape(total_labels)
  gold = gold.reshape(total_labels)

  # f1 scores
  label_f1 = f1_score(gold, predictions, average=None)
  positive_class_index = 1
  print 'f1:', label_f1[positive_class_index]
  print 'all labels:', label_f1
