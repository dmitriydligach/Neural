#!/usr/bin/env python

import numpy
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

class DatasetProvider:
  """THYME relation data"""

  def __init__(self, corpus_path, code_path, max_tokens=10000):
    """Index words by frequency in a file"""

    self.alphabet_file = 'alphabet.txt'
    self.corpus_path = corpus_path
    self.code_path = code_path
    self.max_tokens = max_tokens # max tokens in file

    self.token2int = {} # words indexed by frequency
    self.code2int = {}  # class to int mapping
    self.subj2codes = {} # subj_id to set of icd9 codes

    # making alphabet is expensive so do it once
    if not os.path.isfile(self.alphabet_file):
      print 'making alphabet and writing it to file...'
      self.write_alphabet()
    print 'reading alphabet from file...'
    self.read_alphabet()
    print 'mapping codes...'
    self.map_codes()

  def get_ngrams(self, file_name):
    """Return file as a list of ngrams"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()

    tokens = [] # file as a list of tokens
    for token in text.split():
      if token.isalpha():
        tokens.append(token)

    if len(tokens) > self.max_tokens:
      return None

    ngram_list = []
    for bigram_as_list in nltk.ngrams(tokens, 2):
      ngram_list.append('_'.join(bigram_as_list))
    ngram_list.extend(tokens)

    return ngram_list

  def write_alphabet(self):
    """Write unique corpus tokens to file"""

    # read entire corpus to list!
    tokens = []
    for file in os.listdir(self.corpus_path):
      file_ngram_list = self.get_ngrams(file)
      if file_ngram_list == None:
        continue
      tokens.extend(file_ngram_list)

    # get unique tokens (corpus still in memory!)
    outfile = open(self.alphabet_file, 'w')
    token_counts = collections.Counter(tokens)
    for token, count in token_counts.most_common():
      outfile.write('%s|%s\n' % (token, count))

  def read_alphabet(self, min_tf=100):
    """Read alphabet from file to token2int"""

    index = 1
    self.token2int['oov_word'] = 0
    for line in open(self.alphabet_file):
      token, count = line.strip().split('|')
      if int(count) > min_tf:
        self.token2int[token] = index
        index = index + 1

  def map_codes(self, min_examples_per_code=500):
    """Map subjects to codes and map codes to integers"""

    frame = pandas.read_csv(self.code_path)

    # map subjects to codes first
    for subj_id, icd9_code in zip(frame.SUBJECT_ID, frame.ICD9_CODE):
      if subj_id not in self.subj2codes:
        self.subj2codes[subj_id] = set()
      icd9_category = str(icd9_code)[0:3]
      self.subj2codes[subj_id].add(icd9_category)

    # count code frequencies and write them to file
    code_counter = collections.Counter()
    for codes in self.subj2codes.values():
      code_counter.update(codes)
    outfile = open('codes.txt', 'w')
    for code, count in code_counter.most_common():
      outfile.write('%s|%s\n' % (code, count))

    # make code alphabet for frequent codes
    index = 0
    for code, count in code_counter.most_common():
      if count > min_examples_per_code:
        self.code2int[code] = index
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    codes = []
    examples = []

    for file in os.listdir(self.corpus_path):
      file_ngram_list = self.get_ngrams(file)

      # is this file too long?
      if file_ngram_list == None:
        continue

      # make code vector for this example
      subj_id = int(file.split('.')[0])
      if len(self.subj2codes[subj_id]) == 0:
        print 'skipping file:', file
        continue # no codes for this file

      code_vec = [0] * len(self.code2int)
      for icd9_category in self.subj2codes[subj_id]:
        if icd9_category in self.code2int:
          # this icd9 has enough examples
          code_vec[self.code2int[icd9_category]] = 1
      codes.append(code_vec)

      # make feature vector for this example
      example = []
      for token in set(file_ngram_list):
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]
      examples.append(example)

    return examples, codes

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  code_file = os.path.join(base, cfg.get('data', 'codes'))

  dataset = DatasetProvider(train_dir, code_file)
  print len(dataset.code2int)
