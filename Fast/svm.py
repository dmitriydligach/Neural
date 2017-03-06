#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python -B

import glob, string, ConfigParser, sys, os, numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def load_data(path):
  """Read data from file; return examples and labels"""

  samples = []
  targets = []
  for line in open(path):
    target, text = line.strip().split('|')
    samples.append(text)
    targets.append(target)

  return numpy.array(samples), numpy.array(targets)

def train_and_test(train_file, test_file):
  """Train and test"""

  train_samples, train_targets = load_data(train_file)
  test_samples, test_targets = load_data(test_file)

  vectorizer = CountVectorizer(ngram_range=(1,3))
  train_counts = vectorizer.fit_transform(train_samples)
  test_counts = vectorizer.transform(test_samples)

  tf = TfidfTransformer()
  train_tfidf = tf.fit_transform(train_counts)
  test_tfidf = tf.transform(test_counts)

  classifier = LinearSVC(class_weight='balanced')
  model = classifier.fit(train_tfidf, train_targets)
  predictions = classifier.predict(test_tfidf)

  f1 = f1_score(test_targets, predictions, average=None)
  print 'f1 =', f1

if __name__ == "__main__":

  # settings file specified as command-line argument
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_file = os.path.join(base, cfg.get('data', 'train'))
  test_file = os.path.join(base, cfg.get('data', 'test'))
  print 'train:', train_file
  print 'test:', test_file

  train_and_test(train_file, test_file)
