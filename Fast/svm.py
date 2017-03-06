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
  labels = []
  for line in open(path):
    label, text = line.strip().split('|')
    samples.append(text)
    labels.append(label)

  return numpy.array(samples), numpy.array(labels)

def train_and_test(train_file, test_file):
  """Train and test"""

  samples, labels = load_data(train_file)

  vectorizer = CountVectorizer(
      ngram_range=(1,2),
      min_df=3)
  count_matrix = vectorizer.fit_transform(samples)

  tf = TfidfTransformer()
  tfidf_matrix = tf.fit_transform(count_matrix)

  x_train, x_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels, test_size = 0.1, random_state=0)

  classifier = LinearSVC() # class_weight='balanced')
  model = classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  print 'predictions:', predicted

  precision = precision_score(y_test, predicted, pos_label=1)
  recall = recall_score(y_test, predicted, pos_label=1)
  f1 = f1_score(y_test, predicted, pos_label=1)
  print 'p =', precision
  print 'r =', recall
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
