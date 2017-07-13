#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
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

  classes = len(dataset.label2int)
  maxlen = max([len(seq) for seq in x])
  x = pad_sequences(x, maxlen=maxlen)
  y = to_categorical(y, classes)
  print 'x shape:', x.shape
  print 'y shape:', y.shape
  print 'number of features:', len(dataset.token2int)

  f1_scores = []
  kf = KFold(n_splits=5, shuffle=True, random_state=100)
  for train_indices, test_indices in kf.split(x):

    train_x = x[train_indices]
    train_y = y[train_indices]
    test_x = x[test_indices]
    test_y = y[test_indices]

    model = get_model(cfg, len(dataset.token2int))
    optimizer = RMSprop(lr=cfg.getfloat('nn', 'learnrt'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=cfg.getint('nn', 'epochs'),
              batch_size=cfg.getint('nn', 'batch'),
              validation_split=0.0,
              verbose=0)

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
    f1_scores.append(label_f1[1])

  print 'all f1s:', f1_scores
  print 'average f1:', np.mean(f1_scores)
