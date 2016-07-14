#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python -B

import dataset
import collections
import numpy as np
import sklearn as sk
from sklearn.cross_validation import cross_val_score
import sklearn.svm
 
FOLDS = 2

def balance(x, y, max_examples):
  """Get max_examples of each class"""

  new_x = []
  new_y = []

  for label in set(y):
    count = 0
    for x_i, y_i in zip(x, y):
      if count >= max_examples:
        break
      if y_i == label:
        new_x.append(x_i)
        new_y.append(y_i)
        count = count + 1

  return np.array(new_x), np.array(new_y)

if __name__ == "__main__":

  dataset = dataset.DatasetProvider()
  x, y = dataset.load()
  print 'class distribution:', collections.Counter(y)

  x, y = balance(x, y, 33000)
  print x.shape
  print y.shape
  
  classifier = sk.svm.LinearSVC()
  f1_scores = cross_val_score(classifier, x, y, cv=FOLDS, scoring='f1')
  print 'f1:', f1_scores
  acc_scores = cross_val_score(classifier, x, y, cv=FOLDS)
  print 'acc:', acc_scores
