#!/usr/bin/env python

import numpy
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator
import icd9

def ngrams(text):
  """Generate all unique bigrams from text"""

  ngram_list = []

  for unigram in text.split():
    ngram_list.append(unigram)

  for bigram_as_list in nltk.bigrams(text.split()):
    ngram_list.append('_'.join(bigram_as_list))

  return set(ngram_list)

class DatasetProvider:
  """THYME relation data"""

  def __init__(self, corpus_path, label_path):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.label_path = label_path
    self.token2int = {} # words indexed by frequency
    self.label2int = {} # class to int mapping

  def make_token_alphabet(self, min_df=5, outf='alphabet.txt'):
    """Map tokens to integers and dump to file"""

    tokens = [] # corpus as list
    for file in os.listdir(self.corpus_path):
      text = open(os.path.join(self.corpus_path, file)).read()
      tokens.extend(text.lower().split())

    index = 1 # reserve 0 for oov items
    outfile = open(outf, 'w')
    self.token2int['oov_word'] = 0
    token_counts = collections.Counter(tokens)
    for token, count in token_counts.most_common():
      if count > min_df:
        outfile.write('%s|%s\n' % (token, count))
        self.token2int[token] = index
        index = index + 1

  def make_label_alphabet(self):
    """Map labels to integers"""

    codes = set()
    frame = pandas.read_csv(self.label_path)
    for icd9_code in frame.ICD9_CODE:
      codes.add(icd9_code)

    index = 0
    for code in codes:
      self.label2int[code] = index
      index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    labels = []
    examples = []

    subj2codes = icd9.subject_to_code_map(self.label_path)

    for file in os.listdir(self.corpus_path):
      text = open(os.path.join(self.corpus_path, file)).read()
      example = []
      for token in text.lower().split():
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]
      examples.append(example)

      subj_id = int(file.split('.')[0])
      label_vec = [0] * len(self.label2int)
      for code in subj2codes[subj_id]:
        label_vec[self.label2int[code]] = 1
      labels.append(label_vec)

    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'labels'))

  dataset = DatasetProvider(train_dir, code_file)
  dataset.make_token_alphabet()
  dataset.make_label_alphabet()
  dataset.load()
