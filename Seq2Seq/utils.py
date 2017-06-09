#!/usr/bin/env python

import sys, collections
sys.dont_write_bytecode = True

def make_alphabet(path, field, delimiter='|'):
  """Make alphabet from examples in a file"""

  token2int = {} # token to integer map
  int2token = {} # integer to token map

  token_counts = collections.Counter()
  for line in open(path):
    elements = line.strip().split(delimiter)
    tokens = elements[field].split()
    token_counts.update(tokens)

  index = 1 # starting index
  token2int['oov_word'] = 0
  int2token[0] = 'oov_word'
  for token, _ in token_counts.most_common():
    token2int[token] = index
    int2token[index] = token
    index = index + 1

  return token2int, int2token

def make_alphabet_and_write(path, outfile, field, delimiter='|'):
  """Make alphabet and write to a file"""

  token2int = {} # token to integer map

  token_counts = collections.Counter()
  for line in open(path):
    elements = line.strip().split(delimiter)
    tokens = elements[field].split()
    token_counts.update(tokens)

  outfile = open(outfile, 'w')
  for token, count in token_counts.most_common():
    outfile.write('%s\t%s\n' % (token, count))

def convert_to_int_seq(token_list, token2int, maxlen=float('inf')):
  """Convert list of tokens to a sequence of integers"""

  int_seq = []

  for token in token_list:
    if token in token2int:
      int_seq.append(token2int[token])
    else:
      int_seq.append(token2int['oov_word'])

  # truncate example if it's too long
  if len(int_seq) > maxlen:
    int_seq = int_seq[0:maxlen]

  return int_seq

def convert_to_int_seqs(path, field, token2int,
                              delimiter='|', maxlen=float('inf')):
  """Convert text fragments to sequences of integers"""

  int_seqs = []

  for line in open(path):
    elements = line.strip().split(delimiter)
    token_list = elements[field].split()
    int_seq = convert_to_int_seq(token_list, token2int, maxlen)
    int_seqs.append(int_seq)

  return int_seqs

if __name__ == "__main__":

  token2int, int2token = make_alphabet('temp.txt', 0)
  print 'token2int:', token2int
  print 'int2token:', int2token
  print 'dataset:', convert_to_int_seqs('temp.txt', 0, token2int)
