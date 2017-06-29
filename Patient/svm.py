#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os, numpy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import keras as k
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
import dataset

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))

  # load target task data
  dataset = dataset.DatasetProvider(
    data_dir,
    cfg.get('data', 'alphabet_pickle'))

  x, y = dataset.load()
  # pad to same maxlen as data in source model
  x = pad_sequences(x, maxlen=cfg.getint('data', 'maxlen'))
  print 'x shape (original):', x.shape

  # make vectors for target task
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('ptvec').output)
  x = interm_layer_model.predict(x)
  print 'x shape (new):', x.shape

  # ready for svm train/test now

  if cfg.getfloat('data', 'test_size') == 0:
    # run n-fold cross validation
    classifier = LinearSVC(class_weight='balanced')
    cv_scores = cross_val_score(classifier, x, y, scoring='f1', cv=5)
    print 'cv scores:', cv_scores
    print 'average:', np.mean(cv_scores)

  else:
    # randomly allocate a test set
    x_train, x_test, y_train, y_test = train_test_split(
      x,
      y,
      test_size=cfg.getfloat('data', 'test_size'),
      random_state=1)
    classifier = LinearSVC(class_weight='balanced')
    model = classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    precision = precision_score(y_test, predicted, pos_label=1)
    recall = recall_score(y_test, predicted, pos_label=1)
    f1 = f1_score(y_test, predicted, pos_label=1)
    print 'p =', precision
    print 'r =', recall
    print 'f1 =', f1
