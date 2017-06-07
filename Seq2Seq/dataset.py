#!/usr/bin/env python

import numpy, ConfigParser, os, sys, collections, time
sys.dont_write_bytecode = True
import utils

class DatasetProvider:
  """Corpus for training a language model"""

  def __init__(self, path):
    """Index words by frequency in a file"""

    self.path = path
    self.input2int, self.int2input = \
            utils.make_alphabet(path, '|', 0)
    self.output2int, self.int2output = \
            utils.make_alphabet(path, '|', 1)

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    xs = utils.get_int_seq(self.path, '|', 0, self.input2int)
    ys = utils.get_int_seq(self.path, '|', 1, self.output2int)

    return xs, ys

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))

  dataset = DatasetProvider(train_file)
  examples, labels = dataset.load()

  print dataset.int2output[1]
