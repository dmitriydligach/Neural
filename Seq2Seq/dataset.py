#!/usr/bin/env python

import numpy, ConfigParser, os, sys, collections, time
sys.dont_write_bytecode = True
import utils

class DatasetProvider:
  """Corpus for training a language model"""

  def __init__(self, path):
    """Index words by frequency in a file"""

    self.path = path
    # make alphabet for the input tokens
    self.input2int, self.int2input = utils.make_alphabet(path, 0)
    # make alphabet for the output tokens
    self.output2int, self.int2output = utils.make_alphabet(path, 1)

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    xs = utils.convert_to_int_seqs(self.path, 0, self.input2int)
    ys = utils.convert_to_int_seqs(self.path, 1, self.output2int)

    return xs, ys

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))

  dataset = DatasetProvider(train_file)
  examples, labels = dataset.load()

  utils.make_alphabet_and_write(train_file, 'input_vocabulary.txt', 0)
  utils.make_alphabet_and_write(train_file, 'output_vocabulary.txt', 1)
