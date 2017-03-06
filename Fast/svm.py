#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python -B

import numpy as np
import sklearn as sk
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.cross_validation
import sklearn.svm
from sklearn.datasets.base import Bunch
import glob, string

PATH = '/Users/Dima/Loyola/Data/RtPolarity/rt-polarity.*'
NFOLDS = 10
NGRAMRANGE = (1, 1)
MINDF = 0

def make_bunch():
  """Make a Bunch object from raw data"""
  
  samples = []
  labels = []
  for file_name in glob.glob(PATH):
    print 'reading:', file_name
    with open(file_name) as file:
      for line in file:
        printable = ''.join(c for c in line if c in string.printable)
        samples.append(printable.strip())
        labels.append(file_name.split('.')[1])

  return Bunch(data=np.array(samples), target=np.array(labels))

def run_cross_validation():
  """Run n-fold CV and return average accuracy"""      

  bunch = make_bunch()

  # raw occurences
  vectorizer = sk.feature_extraction.text.CountVectorizer(
    ngram_range=NGRAMRANGE, 
    stop_words=None,
    min_df=MINDF ,
    vocabulary=None,
    binary=False,
    preprocessor=None)
  count_matrix = vectorizer.fit_transform(bunch.data)

  for i in sorted(vectorizer.vocabulary_.values()):
    v = vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)]
    print i, '-', v

  # tf-idf 
  tf = sk.feature_extraction.text.TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)
  
  scores = []
  folds = sk.cross_validation.KFold(len(bunch.data), n_folds=NFOLDS)
  for train_indices, test_indices in folds:
    train_x = tfidf_matrix[train_indices]
    train_y = bunch.target[train_indices]
    test_x = tfidf_matrix[test_indices]
    test_y = bunch.target[test_indices]
    classifier = sk.svm.LinearSVC()
    classifier.fit(train_x, train_y)
    accuracy = classifier.score(test_x, test_y)
    scores.append(accuracy)
  
  print 'accuracy:', np.mean(scores)

if __name__ == "__main__":

  run_cross_validation()
