#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
from sklearn.metrics import f1_score
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
  train_x, train_y = dataset.load()
  maxlen = max([len(seq) for seq in train_x])

  # turn x into numpy array among other things
  classes = len(dataset.code2int)
  train_x = pad_sequences(train_x, maxlen=maxlen)
  train_y = np.array(train_y)
  print 'train_x shape:', train_x.shape
  print 'train_y shape:', train_y.shape
  print 'unique features:', len(dataset.token2int)

  print train_x
  print 'train_x num of elements:', train_x.size
  print 'train_x dtype:', train_x.dtype
  print 'train_x item size in bytes:', train_x.itemsize
  print 'train_x total size in bytes:', train_x.size * train_x.itemsize
  print 'train_x largest value:', np.amax(train_x)

  model = Sequential()
  model.add(Embedding(len(dataset.token2int),
                      cfg.getint('cnn', 'embdims'),
                      input_length=maxlen))
  model.add(GlobalAveragePooling1D())

  model.add(Dense(cfg.getint('cnn', 'hidden')))
  model.add(Activation('relu'))

  model.add(Dense(classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            epochs=cfg.getint('cnn', 'epochs'),
            batch_size=cfg.getint('cnn', 'batch'),
            validation_split=0.1)

  # probability for each class; (test size, num of classes)
  distribution = \
    model.predict(test_x, batch_size=cfg.getint('cnn', 'batch'))
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
