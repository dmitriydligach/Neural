#!/usr/bin/env python

import numpy as np
np.random.seed(1337)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import ConfigParser, os
import sklearn as sk
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import dataset, word2vec
from seq2seq import SimpleSeq2Seq

def print_config(cfg):
  """Print configuration settings"""

  print 'train:', cfg.get('data', 'train')
  print 'test:', cfg.get('data', 'test')
  if cfg.has_option('data', 'embed'):
    print 'embeddings:', cfg.get('data', 'embed')

  print 'batch:', cfg.get('cnn', 'batch')
  print 'epochs:', cfg.get('cnn', 'epochs')
  print 'embdims:', cfg.get('cnn', 'embdims')
  print 'units:', cfg.get('cnn', 'units')
  print 'dropout:', cfg.get('cnn', 'dropout')

# def

if __name__ == "__main__":

  # settings file specified as command-line argument
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  print_config(cfg)
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))

  # learn alphabet from training examples
  dataset = dataset.DatasetProvider(train_file)
  print 'input alphabet size:', len(dataset.input2int)
  print 'output alphabet size:', len(dataset.output2int)

  # now load training examples and labels
  train_x, train_y = dataset.load(train_file)
  maxlen_x = max([len(seq) for seq in train_x])
  maxlen_y = max([len(seq) for seq in train_y])

  # turn x and y into numpy array among other things
  train_x = pad_sequences(train_x, maxlen=maxlen_x)
  train_y = pad_sequences(train_y, maxlen=maxlen_y)
  print train_y.shape
  print train_y

  # convert train_y into (num_examples, maxlen, alphabet_size)
  # train_y = to_categorical(np.array(train_y), classes)

  model = Sequential()
  model.add(Embedding(input_dim=len(dataset.input2int),
                      output_dim=cfg.getint('cnn', 'embdims'),
                      input_length=maxlen_x))
  #model.add(SimpleSeq2Seq(input_dim=5, hidden_dim=10,
  #                        output_length=8, output_dim=8))
  model.add(SimpleSeq2Seq(hidden_dim=10, output_length=maxlen_y, output_dim=len(dataset.output2int)))
  model.compile(loss='mse', optimizer='rmsprop')

  model.fit(train_x,
            train_y,
            epochs=cfg.getint('cnn', 'epochs'),
            batch_size=cfg.getint('cnn', 'batch'),
            verbose=1,
            validation_split=0.0)

  # get_weights() returns a list of 1 element
  # dump these weights to file (word2vec model format)
  weights = model.layers[0].get_weights()[0]
  word2vec.write_vectors(dataset.word2int, weights, 'weights.txt')

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
