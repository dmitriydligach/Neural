#!/usr/bin/env python

import numpy
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

ALPHABET_FILE = '../Codes/alphabet.txt'
MIN_TOKEN_FREQ = 100 # has to be the same for both datasets!

class DatasetProvider:
  """THYME relation data"""

  def __init__(self, corpus_path):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path

    self.token2int = {}
    self.label2int = {'No':0, 'Yes':1}
    self.read_token_alphabet()

  def get_cuis(self, file_name):
    """Return file as a list of CUIs"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()
    tokens = [token for token in text.split()]

    return tokens

  def read_token_alphabet(self):
    """Read alphabet from file to token2int"""

    index = 1
    self.token2int['oov_word'] = 0
    for line in open(ALPHABET_FILE):
      token, count = line.strip().split('|')
      if int(count) > MIN_TOKEN_FREQ:
        self.token2int[token] = index
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    labels = []   # int labels
    examples = [] # int sequence represents each example

    for d in os.listdir(self.corpus_path):
      dir_path = os.path.join(self.corpus_path, d)

      for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        file_feat_list = self.get_cuis(file_path)

        example = []
        for token in set(file_feat_list):
          if token in self.token2int:
            example.append(self.token2int[token])
          else:
            example.append(self.token2int['oov_word'])

        if len(example) > maxlen:
          example = example[0:maxlen]

        examples.append(example)
        labels.append(self.label2int[d])

    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'path'))

  dataset = DatasetProvider(data_dir)
  x, y = dataset.load()
  print y
