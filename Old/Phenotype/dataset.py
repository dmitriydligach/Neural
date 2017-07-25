#!/usr/bin/env python

import numpy, pickle
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

ALPHABET_FILE = 'alphabet.txt'

class DatasetProvider:
  """THYME relation data"""

  def __init__(self,
               corpus_path,
               min_token_freq=0):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.min_token_freq = min_token_freq

    self.label2int = {'No':0, 'Yes':1}
    self.token2int = {}
    self.make_token_alphabet()

  def get_cuis(self, file_name):
    """Return file as a list of CUIs"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()

    # source task trained on no-polarity cuis
    # target task sometimes includes polarity
    # tokens = [token for token in text.split()]
    tokens = []
    for token in text.split():
      if token.startswith('-'):
        tokens.append(token[1:])
      else:
        tokens.append(token)

    return tokens

  def make_token_alphabet(self):
    """Map tokens (CUIs) to integers"""

    # count tokens in the entire corpus
    token_counts = collections.Counter()

    for d in os.listdir(self.corpus_path):
      dir_path = os.path.join(self.corpus_path, d)

      for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        file_feat_list = self.get_cuis(file_path)
        token_counts.update(file_feat_list)

    # now make alphabet
    index = 1
    self.token2int['oov_word'] = 0
    outfile = open(ALPHABET_FILE, 'w')
    for token, count in token_counts.most_common():
      if count > self.min_token_freq:
        outfile.write('%s|%s\n' % (token, count))
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
        # TODO: need to use unique tokens?
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
