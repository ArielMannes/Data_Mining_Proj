import nltk
from nltk.corpus import stopwords
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.utils import shuffle
import string
import pandas as pd


## import dataset
df = pd.read_csv('gender-classifier-DFE-791531.csv', encoding = 'latin1')
df = shuffle(shuffle(shuffle(df)))
##df

## text pre-processing

dist = df.groupby('gender').size().to_frame()


def compute_percentage(x):
    pct = float(x / df['gender'].size) * 100
    return round(pct, 2)

dist['percentage'] = dist.apply(compute_percentage, axis=1)

dist

## TODO : terms frequency for the different genders