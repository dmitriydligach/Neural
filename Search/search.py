#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True

def run():
  """Run grid search"""

  # epochs, num filters, filter len, hidden, dropout, learning rate

  epochs_list = [3,4,5,7,10]
  filters_list = [100, 200, 300]
  filtlen_list = ['2,3', '2,3,4', '2,3,4,5']
  hidden_list = [100, 200, 300]
  dropout_list = [0.25, 0.5]
  learnrt_list = [0.1, 0.001, 0.0001]

  for epochs in epochs_list:
    for filters in filters_list:
      for filtlen in filtlen_list:
        for hidden in hidden_list:
          for dropout in dropout_list:
            for learnrt in learnrt_list:
              run(train_file=train_file,
                  test_file=test_file,
                  batch=batch,
                  epochs=epochs,
                  embdims=embdims,
                  filters=filters,
                  filtlen=filtlen,
                  hidden=hidden,
                  dropout=dropout,
                  learnrt=learnrt)

if __name__ == "__main__":

  run()
