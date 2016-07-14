#!/usr/bin/env python

import numpy as np

import sys
sys.dont_write_bytecode = True

import ConfigParser

import glob, string, collections, operator

class DatasetProvider:
  """THYME relation data"""
  
  def __init__(self, file_names):
    """Index words by frequency in a list of files"""

    self.word2int = {} # words indexed by frequency
    self.label2int = {}   # class to int mapping
    
    unigrams = [] # corpus as list
    labels = []   # classes as list
    for file_name in file_names:
      for line in open(file_name):
        label, text = line.strip().split('|')
        unigrams.extend(text.split())
        labels.append(label)
        
    index = 1 # zero used to encode unknown words
    self.word2int['oov_word'] = 0
    unigram_counts = collections.Counter(unigrams)
    for unigram, count in unigram_counts.most_common():
      self.word2int[unigram] = index
      index = index + 1

    index = 0 # index classes
    for label in set(labels):
      self.label2int[label] = index
      index = index + 1

  def load(self, path):
    """Convert sentences (examples) into lists of indices"""

    examples = []
    labels = []
    for line in open(path):
      label, text = line.strip().split('|')
      example = []
      for unigram in text.split():
        example.append(self.word2int[unigram])
      examples.append(example)
      labels.append(self.label2int[label])

    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])

  dataset = DatasetProvider([cfg.get('data', 'train'),
                             cfg.get('data', 'test')])
  print 'alphabet size:', len(dataset.word2int)

  x,y = dataset.load(cfg.get('data', 'test'))

  print 'max seq len:', max([len(s) for s in x])
  print 'number of examples:', len(x)
  print 'labels:', dataset.label2int
  print 'label counts:', collections.Counter(y)
  print 'first 10 examples:', x[:10]
  print 'class proportions:'
  counter = collections.Counter(y)
  for label in counter:
    print label, counter[label] / float(len(y)), float(len(y)) / counter[label]
