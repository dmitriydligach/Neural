#!/usr/bin/env python

import numpy, ConfigParser, os, sys, collections, time
sys.dont_write_bytecode = True

ALPHABET = 'alphabet.txt' # alphabet file
MINTF = 100 # minimum term frequency in corpus

def make_train_examples(tokens):
  """Make training examples from a token list"""

  # sentence: one two three
  # examples: one|two, one, two|three

  examples = []
  labels = []
  for label_index in range(1, len(tokens)):
    example = tokens[0:label_index]
    label = tokens[label_index]
    examples.append(example)
    labels.append(label)

  return examples, labels

class DatasetProvider:
  """Corpus for training a language model"""

  def __init__(self, path):
    """Index words by frequency in a file"""

    self.path = path # corpus path

    self.token2int = {} # words indexed by frequency
    self.int2token = {} # reversed index

    # Making alphabet is expensive so do it once
    if not os.path.isfile(ALPHABET):
      print 'making alphabet and writing it to file...'
      self.make_alphabet()
    print 'reading alphabet from file...'
    self.read_alphabet()

  def get_tokens(self, sentence):
    """Preprocesses and tokenize a sentence"""

    sentence = sentence.replace('|', '')
    return ['<s>'] + sentence.split() + ['</s>']

  def make_alphabet(self):
    """Write unique corpus tokens to file"""

    token_counts = collections.Counter()
    for line in open(self.path):
      token_counts.update(self.get_tokens(line))

    outfile = open(ALPHABET, 'w')
    for token, count in token_counts.most_common():
      outfile.write('%s|%s\n' % (token, count))

  def read_alphabet(self):
    """Read alphabet from file to token2int"""

    index = 1 # starting index
    self.token2int['oov_word'] = 0
    for line in open(ALPHABET):
      token, count = line.strip().split('|')
      if int(count) > MINTF:
        self.token2int[token] = index
        self.int2token[index] = token
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    xs = []
    ys = []

    for line in open(self.path):

      indices = [] # sentence as list of integers
      for token in self.get_tokens(line):
        if token in self.token2int:
          indices.append(self.token2int[token])
        else:
          indices.append(self.token2int['oov_word'])

      # generate training examples from a sentence
      examples, labels = make_train_examples(indices)

      # attach examples generated from sent to list
      for example in examples:
        if len(example) > maxlen:
          example = example[0:maxlen]
        xs.append(example)
      ys.extend(labels)

    return xs, ys

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))

  t0 = time.time()
  dataset = DatasetProvider(train_file)
  t1 = time.time()
  print 'execute time:', t1 - t0

  examples, labels = dataset.load()
  print len(examples), len(labels)
  print
  print examples
  print
  print labels
