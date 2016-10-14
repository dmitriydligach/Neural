#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import sklearn as sk
from sklearn.metrics import f1_score
import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
import dataset, word2vec_model
import os, ConfigParser

if __name__ == "__main__":
  
  # settings file specified as command-line argument
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))
  print 'train:', train_file
  print 'test:', test_file
  print 'batch:', cfg.get('cnn', 'batch')
  print 'epochs:', cfg.get('cnn', 'epochs')
  print 'embdims:', cfg.get('cnn', 'embdims')
  print 'filters:', cfg.get('cnn', 'filters')
  print 'filtlen:', cfg.get('cnn', 'filtlen')
  print 'hidden:', cfg.get('cnn', 'hidden')
  print 'dropout:', cfg.get('cnn', 'dropout')
  print 'learnrt:', cfg.get('cnn', 'learnrt')

  # learn alphabets from training examples
  dataset = dataset.DatasetProvider(train_file)
  # now load training examples and labels
  train_left, train_middle, train_right, train_y = dataset.load(train_file)
  left_maxlen = max([len(seq) for seq in train_left])
  middle_maxlen = max([len(seq) for seq in train_middle])
  right_maxlen = max([len(seq) for seq in train_right])

  # now load test examples and labels
  test_left, test_middle, test_right, test_y = \
    dataset.load(test_file, left_maxlen=left_maxlen,
                 middle_maxlen=middle_maxlen, right_maxlen=right_maxlen)
  
  # turn x and y into numpy array among other things
  classes = len(set(train_y))
  train_left = pad_sequences(train_left, maxlen=left_maxlen)
  train_middle = pad_sequences(train_middle, maxlen=middle_maxlen)
  train_right = pad_sequences(train_right, maxlen=right_maxlen)
  train_y = to_categorical(np.array(train_y), classes)  
  test_left = pad_sequences(test_left, maxlen=left_maxlen)
  test_middle = pad_sequences(test_middle, maxlen=middle_maxlen)
  test_right = pad_sequences(test_right, maxlen=right_maxlen)
  test_y = to_categorical(np.array(test_y), classes)  

  print 'train_left shape:', train_left.shape
  print 'train_middle shape:', train_middle.shape
  print 'train_right shape:', train_right.shape
  print 'train_y shape:', train_y.shape
  print 'test_left shape:', test_left.shape
  print 'test_middle shape:', test_middle.shape
  print 'test_right shape:', test_right.shape
  print 'test_y shape:', test_y.shape, '\n'

  branches = [] # models to be merged
  train_xs = [] # train x for each branch 
  test_xs = []  # test x for each branch

  for filter_len in cfg.get('cnn', 'filtlen').split(','):

    branch = Sequential()
    branch.add(Embedding(input_dim=len(dataset.left2int),
                         output_dim=cfg.getint('cnn', 'embdims'),
                         input_length=left_maxlen))
    branch.add(Convolution1D(nb_filter=cfg.getint('cnn', 'filters'),
                             filter_length=int(filter_len),
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    branch.add(MaxPooling1D(pool_length=2))
    branch.add(Flatten())

    branches.append(branch)
    train_xs.append(train_left)
    test_xs.append(test_left)

  for filter_len in cfg.get('cnn', 'filtlen').split(','):

    branch = Sequential()
    branch.add(Embedding(input_dim=len(dataset.middle2int),
                         output_dim=cfg.getint('cnn', 'embdims'),
                         input_length=middle_maxlen))
    branch.add(Convolution1D(nb_filter=cfg.getint('cnn', 'filters'),
                             filter_length=int(filter_len),
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    branch.add(MaxPooling1D(pool_length=2))
    branch.add(Flatten())

    branches.append(branch)
    train_xs.append(train_middle)
    test_xs.append(test_middle)

  for filter_len in cfg.get('cnn', 'filtlen').split(','):

    branch = Sequential()
    branch.add(Embedding(input_dim=len(dataset.right2int),
                         output_dim=cfg.getint('cnn', 'embdims'),
                         input_length=right_maxlen))
    branch.add(Convolution1D(nb_filter=cfg.getint('cnn', 'filters'),
                             filter_length=int(filter_len),
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    branch.add(MaxPooling1D(pool_length=2))
    branch.add(Flatten())

    branches.append(branch)
    train_xs.append(train_right)
    test_xs.append(test_right)

  model = Sequential()
  model.add(Merge(branches, mode='concat'))

  model.add(Dropout(cfg.getfloat('cnn', 'dropout')))
  model.add(Dense(cfg.getint('cnn', 'hidden')))
  model.add(Activation('relu'))

  model.add(Dropout(cfg.getfloat('cnn', 'dropout')))
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  optimizer = RMSprop(lr=cfg.getfloat('cnn', 'learnrt'),
                      rho=0.9, epsilon=1e-08)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(train_xs,
            train_y,
            nb_epoch=cfg.getint('cnn', 'epochs'),
            batch_size=cfg.getint('cnn', 'batch'),
            verbose=1,
            validation_split=0.1,
            class_weight=None)

  # probability for each class; (test size, num of classes)
  distribution = \
    model.predict(test_xs, batch_size=cfg.getint('cnn', 'batch'))
  # class predictions; (test size,)
  predictions = np.argmax(distribution, axis=1)
  # gold labels; (test size,)
  gold = np.argmax(test_y, axis=1)

  # f1 scores
  label_f1 = f1_score(gold, predictions, average=None)

  print
  for label, idx in dataset.label2int.items():
    print 'f1(%s)=%f' % (label, label_f1[idx])

  if 'contains' in dataset.label2int:
    idxs = [dataset.label2int['contains'], dataset.label2int['contains-1']]
    contains_f1 = f1_score(gold, predictions, labels=idxs, average='micro')
    print '\nf1(contains average) =', contains_f1
  else:
    idxs = dataset.label2int.values()
    average_f1 = f1_score(gold, predictions, labels=idxs, average='micro')
    print 'f1(all) =', average_f1
