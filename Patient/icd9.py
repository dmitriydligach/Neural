#! /usr/bin/env python
import pandas, string, os

DIAG_ICD9 = '/Users/Dima/Loyola/Data/MimicIII/Source/DIAGNOSES_ICD.csv'
PROC_ICD9 = '/Users/Dima/Loyola/Data/MimicIII/Source/PROCEDURES_ICD.csv'
OUTFILE = '/Users/Dima/Loyola/Data/MimicIII/PatientVec/'

def subject_to_code_map(path):
  """Dictionary mapping subject ids to icd9 codes"""

  # read data frame from CSV file
  frame = pandas.read_csv(path)

  subj2codes = {} # key: subj_id, value: set of icd9 codes
  for subj_id, icd9_code in zip(frame.SUBJECT_ID, frame.ICD9_CODE):
    if subj_id not in subj2codes:
      subj2codes[subj_id] = set()
    subj2codes[subj_id].add(icd9_code)

  return subj2codes

if __name__ == "__main__":

  subj2codes = subject_to_code_map(PROC_ICD9)
  print subj2codes[45]
