#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os, numpy
import keras as k
import sklearn as sk
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Model
import dataset

MODEL_PATH = '../Codes/model.h5'
MAXLEN = 1387 # must be same as in source task

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))

  dataset = dataset.DatasetProvider(data_dir)
  x, y = dataset.load()
  x = pad_sequences(x, maxlen=MAXLEN)
  print 'source x shape:', x.shape
  print 'y len:', len(y)

  # make vectors for target task
  model = load_model(MODEL_PATH)
  interm_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('ptvec').output)
  x = interm_layer_model.predict(x)
  print 'target x shape:', x.shape

  # conduct svm training
  x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=1)
  classifier = sk.svm.LinearSVC(class_weight='balanced')
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  f1 = f1_score(y_test, predicted, pos_label=1)
  print 'f1 =', f1
