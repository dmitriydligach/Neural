#!/usr/bin/env python

import numpy as np
import sys, ConfigParser, collections, os
sys.dont_write_bytecode = True

class DatasetProvider:
  """THYME relation data"""
    
  def __init__(self, path):
    """Make alphabets"""

    # various alphabets
    self.label2int = {}
    self.left2int = {}
    self.larg2int = {}
    self.middle2int = {}
    self.rarg2int = {}
    self.right2int = {}

    labels = []  # classes as list
    lefts = []   # left regions
    largs = []   # left arg regions
    middles = [] # middle regions
    rargs = []   # right arg regions
    rights = []  # right regions
    
    for line in open(path):
      label, left, larg, middle, rarg, right = line.strip().split('|')
      labels.append(label)
      lefts.extend(left.split())
      largs.extend(larg.split())
      middles.extend(middle.split())
      rargs.extend(rarg.split())
      rights.extend(right.split())

    self.left2int = make_alphabet(lefts)
    self.larg2int = make_alphabet(largs)
    self.middle2int = make_alphabet(middles)
    self.rarg2int = make_alphabet(rargs)
    self.right2int = make_alphabet(rights)

    index = 0 # index classes
    for label in set(labels):
      self.label2int[label] = index
      index = index + 1
      
  def load(self, path, left_maxlen=float('inf'), larg_maxlen=float('inf'),
        middle_maxlen=float('inf'), rarg_maxlen=float('inf'), right_maxlen=float('inf')):
    """Convert sentences (examples) into lists of indices"""

    # lists of int sequences
    lefts = []
    rargs = []
    middles = []
    largs = []
    rights = []
    labels = []
    
    for line in open(path):
      label, left, larg, middle, rarg, right = line.strip().split('|')

      lefts.append(convert_to_ints(left, self.left2int, left_maxlen))
      largs.append(convert_to_ints(larg, self.larg2int, larg_maxlen))
      middles.append(convert_to_ints(middle, self.middle2int, middle_maxlen))
      rargs.append(convert_to_ints(rarg, self.rarg2int, rarg_maxlen))
      rights.append(convert_to_ints(right, self.right2int, right_maxlen))
      labels.append(self.label2int[label])

    return lefts, largs, middles, rargs, rights, labels

def make_alphabet(tokens):
  """Map tokens to integers sorted by frequency"""
  
  token2int = {} # key: token, value: int
  index = 1 # start from 1 (zero reserved)
  token2int['oov_word'] = 0
  
  # tokens will be indexed by frequency
  counts = collections.Counter(tokens)
  for token, count in counts.most_common():
    token2int[token] = index
    index = index + 1

  return token2int
  
def convert_to_ints(text, alphabet, maxlen=float('inf')):
  """Turn text into a sequence of integers"""
  
  result = []
  for token in text.split():
    if token in alphabet:
      result.append(alphabet[token])
    else:
      result.append(alphabet['oov_word'])
      
  if len(result) > maxlen:
    return result[0:maxlen]
  else:
    return result
  
if __name__ == "__main__":
  
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))

  dataset = DatasetProvider(train_file)
  l, la, m, ra, r, label = dataset.load(test_file)
  print 'first 10 examples:', la[:50]

  l = ['one', 'two', 'three', 'one', 'four', 'three', 'one']
  print 'corpus:', l
  alphabet = make_alphabet(l)
  print 'alphabet:', alphabet
  print 'int sequence:', convert_to_ints('one two', alphabet)
  
