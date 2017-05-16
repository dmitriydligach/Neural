#!/usr/bin/env python

import numpy
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

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

  def __init__(self, corpus_path, code_path, maxsize=10000):
    """Index words by frequency in a file"""

    self.alphabet_file = 'alphabet.txt'
    self.corpus_path = corpus_path
    self.code_path = code_path
    self.max_tokens = maxsize # max tokens in file

    self.token2int = {} # words indexed by frequency
    self.code2int = {}  # class to int mapping
    self.subj2codes = {} # subj_id to set of icd9 codes

    # making alphabet is expensive so do it once
    if not os.path.isfile(self.alphabet_file):
      self.write_alphabet()
    self.read_alphabet()
    self.make_code_alphabet()

  def get_tokens(self, file_name):
    """Return file as a list of tokens"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()

    tokens = [] # file as a list of tokens
    for token in text.split():
      if token.isalpha():
        # token = token.replace('|', '')
        tokens.append(token)

    if len(tokens) > self.max_tokens:
      return []
    else:
      return tokens

  def write_alphabet(self):
    """Write unique corpus tokens to file"""

    tokens = [] # corpus as a list of tokens
    for file in os.listdir(self.corpus_path):
      file_tokens = self.get_tokens(file)
      tokens.extend(file_tokens)

    outfile = open(self.alphabet_file, 'w')
    token_counts = collections.Counter(tokens)
    for token, count in token_counts.most_common():
      outfile.write('%s|%s\n' % (token, count))

  def read_alphabet(self, mintf=10):
    """Read alphabet from file to token2int"""

    index = 1
    self.token2int['oov_word'] = 0
    for line in open(self.alphabet_file):
      token, count = line.strip().split('|')
      if int(count) > mintf:
        self.token2int[token] = index
        index = index + 1

  def make_code_alphabet(self):
    """Map codes to integers"""

    # category: first three digits
    # subcategory: fourth digit
    # subclassification: fifth digit
    # for now only using icd9 categories

    codes = set()
    frame = pandas.read_csv(self.code_path)
    for icd9_code in frame.ICD9_CODE:
      icd9_category = str(icd9_code)[0:3]
      codes.add(icd9_category)

    index = 0
    for code in codes:
      self.code2int[code] = index
      index = index + 1

  def map_subject_to_codes(self):
    """Dictionary mapping subject ids to icd9 codes"""

    frame = pandas.read_csv(self.code_path)

    for subj_id, icd9_code in zip(frame.SUBJECT_ID, frame.ICD9_CODE):
      if subj_id not in self.subj2codes:
        self.subj2codes[subj_id] = set()
      self.subj2codes[subj_id].add(str(icd9_code))

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    self.map_subject_to_codes()

    codes = []
    examples = []
    for file in os.listdir(self.corpus_path):
      file_tokens = self.get_tokens(file)

      example = []
      for token in file_tokens:
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]
      examples.append(example)

      subj_id = int(file.split('.')[0])
      code_vec = [0] * len(self.code2int)
      for icd9_code in self.subj2codes[subj_id]:
        icd9_category = icd9_code[0:3]
        code_vec[self.code2int[icd9_category]] = 1
      codes.append(code_vec)

    return examples, codes

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))

  dataset = DatasetProvider(train_dir, code_file)
  print len(dataset.code2int)
