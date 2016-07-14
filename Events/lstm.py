#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python -B

import numpy as np
import sklearn as sk
import sklearn.cross_validation
import keras as k
import keras.utils.np_utils
import dataset

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU

NFOLDS = 10
BATCH = 50
EPOCHS = 5
EMBDIMS = 300

if __name__ == "__main__":

  dataset = dataset.DatasetProvider()
  x, y = dataset.load()
  print 'x shape:', x.shape
  print 'y shape:', y.shape
  
  scores = []
  folds = sk.cross_validation.KFold(len(y), n_folds=NFOLDS, shuffle=True)

  for fold_num, (train_indices, test_indices) in enumerate(folds):
    train_x = x[train_indices]
    train_y = y[train_indices]
    test_x = x[test_indices]
    test_y = y[test_indices]
    
    model = k.models.Sequential()
    model.add(LSTM(128, input_length=205845, input_dim=300))
    # model.add(Dense(128, input_shape=(EMBDIMS,)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              nb_epoch=EPOCHS,
              batch_size=BATCH,
              verbose=0,
              validation_split=0.1)
    score, accuracy = model.evaluate(test_x,
                                     test_y,
                                     batch_size=BATCH,
                                     verbose=0)
    # todo: what is score?
    print 'fold %d accuracy: %f' % (fold_num, accuracy)
    scores.append(accuracy)
  
  print np.mean(scores)
