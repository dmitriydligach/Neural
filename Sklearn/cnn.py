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
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import dataset
import word2vec

global cfg

def print_config(cfg):
  """Print configuration settings"""

  print 'train:', cfg.get('data', 'train')
  print 'test:', cfg.get('data', 'test')
  if cfg.has_option('data', 'embed'):
    print 'embeddings:', cfg.get('data', 'embed')

  print 'batch:', cfg.get('cnn', 'batch')
  print 'embdims:', cfg.get('cnn', 'embdims')
  print 'filters:', cfg.get('cnn', 'filters')
  print 'learnrt:', cfg.get('cnn', 'learnrt')

def make_model(kernel_size, hidden_size, dropout):
  """Creating a model for sklearn"""

  print '\n'
  print 'kernel_size:', kernel_size
  print 'hidden_size:', hidden_size
  print 'dropout:', dropout
  print

  init_vectors = None
  if cfg.has_option('data', 'embed'):
    embed_file = os.path.join(base, cfg.get('data', 'embed'))
    w2v = word2vec.Model(embed_file)
    init_vectors = [w2v.select_vectors(dataset.word2int)]

  model = Sequential()
  model.add(Embedding(len(dataset.word2int),
                      cfg.getint('cnn', 'embdims'),
                      input_length=maxlen,
                      trainable=True,
                      weights=init_vectors))
  model.add(Conv1D(filters=cfg.getint('cnn', 'filters'),
                   kernel_size=kernel_size,
                   activation='relu'))
  model.add(GlobalMaxPooling1D())

  model.add(Dropout(dropout))
  model.add(Dense(hidden_size))
  model.add(Activation('relu'))

  model.add(Dropout(dropout))
  model.add(Dense(classes))
  model.add(Activation('softmax'))

  optimizer = RMSprop(lr=cfg.getfloat('cnn', 'learnrt'),
                      rho=0.9, epsilon=1e-08)
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

  return model

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))

  dataset = dataset.DatasetProvider(train_file)
  train_x, train_y = dataset.load(train_file)
  maxlen = max([len(seq) for seq in train_x])
  test_x, test_y = dataset.load(test_file, maxlen=maxlen)

  classes = len(set(train_y))
  train_x = pad_sequences(train_x, maxlen=maxlen)
  train_y = to_categorical(np.array(train_y), classes)
  test_x = pad_sequences(test_x, maxlen=maxlen)
  test_y = to_categorical(np.array(test_y), classes)

  classifier = KerasClassifier(make_model, batch_size=50)
  validator = GridSearchCV(classifier,
                           param_grid={'kernel_size': [2,3],
                                       'hidden_size': [100,200,500],
                                       'dropout': [0.25,10.5],
                                       'epochs': [3,4,5]},
                           scoring='log_loss',
                           cv=2)
  validator.fit(train_x, train_y)
  print('The parameters of the best model are: ')
  print(validator.best_params_)
  best_model = validator.best_estimator_.model

  # probability for each class; (test size, num of classes)
  distribution = \
    best_model.predict(test_x, batch_size=cfg.getint('cnn', 'batch'))
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
