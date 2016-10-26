#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
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
import dataset
import word2vec_model

def run(train_file,
         test_file,
         batch,
         epochs,
         embdims,
         filters,
         filtlen,
         hidden,
         dropout,
         learnrt):
  """Train/test with given parameters. Return F1."""

  print 'train:', train_file
  print 'test:', test_file
  print 'batch:', batch
  print 'epochs:', epochs
  print 'embdims:', embdims
  print 'filters:', filters
  print 'filtlen:', filtlen
  print 'hidden:', hidden
  print 'dropout:', dropout
  print 'learnrt:', learnrt
    
  # learn alphabet from training examples
  datset = dataset.DatasetProvider(train_file)
  # now load training examples and labels
  train_x, train_y = datset.load(train_file)
  maxlen = max([len(seq) for seq in train_x])
  # now load test examples and labels
  test_x, test_y = datset.load(test_file, maxlen=maxlen)
  
  # turn x and y into numpy array among other things
  classes = len(set(train_y))
  train_x = pad_sequences(train_x, maxlen=maxlen)
  train_y = to_categorical(np.array(train_y), classes)  
  test_x = pad_sequences(test_x, maxlen=maxlen)
  test_y = to_categorical(np.array(test_y), classes)  

  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'test_x shape:', test_x.shape
  print 'test_y shape:', test_y.shape, '\n'

  branches = [] # models to be merged
  train_xs = [] # train x for each branch 
  test_xs = []  # test x for each branch
  
  for filter_len in filtlen.split(','):

    branch = Sequential()
    branch.add(Embedding(len(datset.word2int),
                         embdims,
                         input_length=maxlen))
    branch.add(Convolution1D(nb_filter=filters,
                             filter_length=int(filter_len),
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    branch.add(MaxPooling1D(pool_length=2))
    branch.add(Flatten())

    branches.append(branch)
    train_xs.append(train_x)
    test_xs.append(test_x)

  model = Sequential()
  model.add(Merge(branches, mode='concat'))
  
  model.add(Dropout(dropout))
  model.add(Dense(hidden))
  model.add(Activation('relu'))

  model.add(Dropout(dropout))
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  optimizer = RMSprop(lr=learnrt,
                      rho=0.9, epsilon=1e-08)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(train_xs,
            train_y,
            nb_epoch=epochs,
            batch_size=batch,
            verbose=1,
            validation_split=0.1,
            class_weight=None)

  # probability for each class; (test size, num of classes)
  distribution = \
    model.predict(test_xs, batch_size=batch)
  # class predictions; (test size,)
  predictions = np.argmax(distribution, axis=1)
  # gold labels; (test size,)
  gold = np.argmax(test_y, axis=1)

  # f1 scores
  label_f1 = f1_score(gold, predictions, average=None)

  print
  for label, idx in datset.label2int.items():
    print 'f1(%s)=%f' % (label, label_f1[idx])

  if 'contains' in datset.label2int:
    idxs = [datset.label2int['contains'], datset.label2int['contains-1']]
    contains_f1 = f1_score(gold, predictions, labels=idxs, average='micro')
    print '\nf1(contains average) =', contains_f1
  else:
    idxs = datset.label2int.values()
    average_f1 = f1_score(gold, predictions, labels=idxs, average='micro')
    print 'f1(all) =', average_f1

  print '\n***************************************************************\n'
  
if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  config_file = sys.argv[1]
  cfg.read(config_file)
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))
  batch = cfg.getint('cnn', 'batch')
  embdims = cfg.getint('cnn', 'embdims')
  
  if len(sys.argv) == 2:
    run(train_file=train_file,
        test_file=test_file,
        batch=batch,
        epochs=cfg.getint('cnn', 'epochs'),
        embdims=embdims,
        filters=cfg.getint('cnn', 'filters'),
        filtlen=cfg.get('cnn', 'filtlen'),
        hidden=cfg.getint('cnn', 'hidden'),
        dropout=cfg.getfloat('cnn', 'dropout'),
        learnrt=cfg.getfloat('cnn', 'learnrt'))

  else:
    epochs_list = [3,4,5]
    filters_list = [100, 200, 300]
    filtlen_list = ['2,3', '2,3,4', '2,3,4,5']
    hidden_list = [100, 200, 300]
    dropout_list = [0.25, 0.5]
    learnrt_list = [0.1, 0.001, 0.0001]
  
    for epochs in epochs_list:
      for filters in filters_list:
        for filtlen in filtlen_list:
          for hidden in hidden_list:
            for dropout in dropout_list:
              for learnrt in learnrt_list:
                run(train_file=train_file,
                    test_file=test_file,
                    batch=batch,
                    epochs=epochs,
                    embdims=embdims,
                    filters=filters,
                    filtlen=filtlen,
                    hidden=hidden,
                    dropout=dropout,
                    learnrt=learnrt)
