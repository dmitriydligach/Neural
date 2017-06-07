#!/usr/bin/env python

import sys, collections
sys.dont_write_bytecode = True

def make_alphabet(path, delimiter, field):
  """Make alphabet from examples in a file"""

  token2int = {}
  int2token = {}

  token_counts = collections.Counter()
  for line in open(path):
    elements = line.strip().split(delimiter)
    tokens = elements[field].split()
    token_counts.update(tokens)

  index = 1 # starting index
  token2int['oov_word'] = 0
  int2token[0] = 'oov_word'
  for token, count in token_counts.most_common():
    token2int[token] = index
    int2token[index] = token
    index = index + 1

  return token2int, int2token

def get_int_seq(path, delimiter, field,
                  token2int, maxlen=float('inf')):
  """Convert text fragments to int sequences"""

  examples = []

  for line in open(path):
    elements = line.strip().split('|')

    example = []
    for token in elements[field].split():
      if token in token2int:
        example.append(token2int[token])
      else:
        example.append(token2int['oov_word'])

    # truncate example if it's too long
    if len(example) > maxlen:
      example = example[0:maxlen]

    examples.append(example)

  return examples

if __name__ == "__main__":

  token2int = make_alphabet('temp.txt', '|', 0)
  print token2int
  print
  print make_train_examples('temp.txt', '|', 0, token2int)
