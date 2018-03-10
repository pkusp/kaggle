import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

train = pd.read_csv('../../data_raw/train.csv')
test = pd.read_csv('../../data_raw/test.csv')
subm = pd.read_csv('../../data_raw/sample_submission.csv')

print(train.head())

