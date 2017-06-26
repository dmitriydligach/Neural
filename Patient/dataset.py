#!/usr/bin/env python

import numpy
import ConfigParser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

ALPHABET_FILE = 'alphabet.txt'
CODE_FREQ_FILE = 'codes.txt'
DIAG_ICD9_FILE = 'DIAGNOSES_ICD.csv'
PROC_ICD9_FILE = 'PROCEDURES_ICD.csv'
CPT_CODE_FILE = 'CPTEVENTS.csv'
MIN_TOKEN_FREQ = 100
MAX_TOKENS_IN_FILE = 10000
MIN_EXAMPLES_PER_CODE = 500

class DatasetProvider:
  """THYME relation data"""

  def __init__(self, corpus_path, code_dir):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.code_dir = code_dir

    self.token2int = {}  # words indexed by frequency
    self.code2int = {}   # class to int mapping
    self.subj2codes = {} # subj_id to set of icd9 codes

    # making token alphabet is expensive so do it once
    if not os.path.isfile(ALPHABET_FILE):
      print 'making alphabet and writing it to file...'
      self.make_and_write_token_alphabet()
    print 'reading alphabet from file...'
    self.read_token_alphabet()
    print 'mapping codes...'
    diag_code_file = os.path.join(self.code_dir, DIAG_ICD9_FILE)
    proc_code_file = os.path.join(self.code_dir, PROC_ICD9_FILE)
    cpt_code_file = os.path.join(self.code_dir, CPT_CODE_FILE)
    self.map_subjects_to_codes(diag_code_file, 'ICD9_CODE', 'diag', 3)
    self.map_subjects_to_codes(proc_code_file, 'ICD9_CODE', 'proc', 2)
    self.map_subjects_to_codes(cpt_code_file, 'CPT_NUMBER', 'cpt', 5)
    self.make_code_alphabet()

  def get_ngrams(self, file_name):
    """Return file as a list of ngrams"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()

    tokens = [] # file as a list of tokens
    for token in text.split():
      if token.isalpha():
        tokens.append(token)

    if len(tokens) > MAX_TOKENS_IN_FILE:
      return None

    ngram_list = []
    for bigram_as_list in nltk.ngrams(tokens, 2):
      ngram_list.append('_'.join(bigram_as_list))
    ngram_list.extend(tokens)

    return ngram_list

  def get_cuis(self, file_name):
    """Return file as a list of CUIs"""

    infile = os.path.join(self.corpus_path, file_name)
    text = open(infile).read().lower()
    tokens = [token for token in text.split()]
    if len(tokens) > MAX_TOKENS_IN_FILE:
      return None

    return tokens

  def make_and_write_token_alphabet(self):
    """Write unique corpus tokens to file"""

    # read entire corpus to list!
    tokens = []
    for file in os.listdir(self.corpus_path):
      file_ngram_list = self.get_cuis(file)
      if file_ngram_list == None:
        continue
      tokens.extend(file_ngram_list)

    # get unique tokens (corpus still in memory!)
    outfile = open(ALPHABET_FILE, 'w')
    token_counts = collections.Counter(tokens)
    for token, count in token_counts.most_common():
      outfile.write('%s|%s\n' % (token, count))

  def read_token_alphabet(self):
    """Read alphabet from file to token2int"""

    index = 1
    self.token2int['oov_word'] = 0
    for line in open(ALPHABET_FILE):
      token, count = line.strip().split('|')
      if int(count) > MIN_TOKEN_FREQ:
        self.token2int[token] = index
        index = index + 1

  def map_subjects_to_codes(self,
                            code_file,
                            code_col,
                            prefix,
                            num_digits):
    """Map subjects to codes"""

    frame = pandas.read_csv(code_file)

    for subj_id, code in zip(frame.SUBJECT_ID, frame[code_col]):
      if subj_id not in self.subj2codes:
        self.subj2codes[subj_id] = set()
      short_code = '%s_%s' % (prefix, str(code)[0:num_digits])
      self.subj2codes[subj_id].add(short_code)

  def make_code_alphabet(self):
    """Map codes to integers"""

    # count code frequencies and write them to file
    code_counter = collections.Counter()
    for codes in self.subj2codes.values():
      code_counter.update(codes)
    outfile = open(CODE_FREQ_FILE, 'w')
    for code, count in code_counter.most_common():
      outfile.write('%s|%s\n' % (code, count))

    # make code alphabet for frequent codes
    index = 0
    for code, count in code_counter.most_common():
      if count > MIN_EXAMPLES_PER_CODE:
        self.code2int[code] = index
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    codes = []    # each example has multiple codes
    examples = [] # int sequence represents each example

    for file in os.listdir(self.corpus_path):
      file_ngram_list = self.get_cuis(file)
      if file_ngram_list == None:
        continue # file too long

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

      if sum(code_vec) == 0:
        continue # all rare codes for this file

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
