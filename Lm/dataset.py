#!/usr/bin/env python

import numpy, ConfigParser, os, sys, collections, time
sys.dont_write_bytecode = True

ALPHABET = 'alphabet.txt' # alphabet file
MINTF = 100 # minimum term frequency in corpus

def make_training_examples(sentence):
  """Make training examples from a sentence"""

  # sentence: one two three
  # examples: one|two, one, two|three

  examples = []
  for label_index in range(1, len(sentence)):
    example = sentence[0:label_index]
    label = sentence[label_index]
    print example, " -> ", label

class DatasetProvider:
  """Corpus for training a language model"""

  def __init__(self, path):
    """Index words by frequency in a file"""

    self.path = path # corpus path

    self.token2int = {} # words indexed by frequency
    self.int2token = {} # reversed index

    # making alphabet is expensive so do it once
    if not os.path.isfile(ALPHABET):
      print 'making alphabet and writing it to file...'
      self.make_alphabet()
    print 'reading alphabet from file...'
    self.read_alphabet()

  def make_alphabet(self):
    """Write unique corpus tokens to file"""

    # read entire corpus to list!
    corpus_tokens = []
    for sentence in open(self.path):
      # sentence_tokens = ['<s>'] + sentence.split() + ['</s>']
      sentence_tokens = sentence.split()
      corpus_tokens.extend(sentence_tokens)

    # get unique tokens (corpus still in memory!)
    outfile = open(ALPHABET, 'w')
    token_counts = collections.Counter(corpus_tokens)
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
        index = index + 1

  def load(self, maxlen=float('inf')):
    """Convert examples into lists of indices"""

    examples = []
    for sentence in open(self.path):
      sentence_tokens = ['<s>'] + sentence.split() + ['</s>']
      sentence_indices = []
      for token in sentence_tokens:
        if token in self.token2int:
          sentence_indices.append(self.token2int[token])
        else:
          sentence_indices.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]
      examples.append(example)

    return examples, codes

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))


  t0 = time.time()
  dataset = DatasetProvider(train_file)
  t1 = time.time()
  print 'execute time:', t1 - t0
