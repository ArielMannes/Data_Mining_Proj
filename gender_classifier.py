import nltk
from nltk.corpus import stopwords
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import shuffle
from nltk.probability import FreqDist
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from zmq.backend.cython import message

ps = PorterStemmer()
stop_words = stopwords.words('english') + list(string.punctuation)

## import dataset
df = pd.read_csv('gender-classifier-DFE-791531.csv', encoding='latin1')
df = shuffle(shuffle(shuffle(df)))
df

## Question 1


## text pre-processing
dist = df.groupby('gender').size().to_frame()


# compute gender disterbution before text processing
def compute_percentage(x):
    pct = float(x / df['gender'].size) * 100
    return round(pct, 2)


dist['percentage'] = dist.apply(compute_percentage, axis=1)
dist

# remove points we're not confident about
filtered = df.values.tolist()
filtered = list(filter(lambda x: x[6] == 1.0, filtered))

# split our data to three smaller lists by gender
males = list(filter(lambda x: x[5] == "male", filtered))
females = list(filter(lambda x: x[5] == "female", filtered))
brands = list(filter(lambda x: x[5] == "brand", filtered))


def try_str(t):
    try:
        return str(t)
    except:
        return ""

        # remove characters like "'" or "."


def wordify(t):
    t = t.replace("'", "")
    t = t.replace(",", " ")
    t = t.replace(".", " ")
    t = t.replace("!", " ")
    t = t.replace("?", " ")
    t = t.replace("&", " ")
    t = t.replace("|", " ")
    t = t.replace("/", "")
    t = t.replace(";", " ")
    t = t.replace(":", " ")
    t = t.replace("\\", "")
    t = t.replace("\\n", " ")
    return t


# combine descriptions and text tweets into a single list of 'words'
def get_list_of_words(lst):
    word_list = " ".join(list(map(lambda x: try_str(x[10]), lst)))
    word_list = " ".join(list(map(lambda x: try_str(x[19]), lst)))
    word_list = word_list.lower().split(" ")
    # word_list = wordify(word_list).split(" ")
    word_list = list(w for w in word_list if len(w) > 1 and w not in stop_words)
    return word_list


def word_distrebution(text, messeg):
    ret_list = []
    list_words = " ".join(list(map(lambda x: try_str(x[10])+try_str(x[19]), text)))
    #list_words = " ".join(list(map(lambda x: try_str(x[19]), text)))
    list_words = list_words.lower().split(" ")
    # word_list = wordify(word_list).split(" ")
    list_words = list(w for w in list_words if len(w) > 1 and w not in stop_words)
    #list_words = get_list_of_words(text)
    word_dist = FreqDist(list_words)
    feature_set = word_dist.most_common(4000)
    word_dist = word_dist.most_common(10)
    dist = pd.DataFrame.from_records(word_dist).transpose
    print (messeg)
    print dist

    for word in feature_set:
        ret_list.append(word[0])

    return  ret_list



male_features = word_distrebution(males, 'Male most frequent words')

print ()
female_features = word_distrebution(females, 'Female most frequent words')

print ()
brand_features = word_distrebution(brands, 'Brand most frequent words')

## Question 2
def find_features(top_words, text):
    feature = {}
    for word in top_words:
        feature[word] = word in text.lower()
    return feature

def union(a, b, c):
    first = (set(a) | set(b))
    return list(first | set(c))

top_words = union(male_features, female_features, brand_features)
tweet_by_gender = (map(lambda x: (try_str(x[10])+ try_str(x[19]), x[5]),filtered))

feature_set = [(find_features(top_words, line[0]), line[1]) for line in tweet_by_gender]
training_set = feature_set[:int(len(feature_set)*4/5)]
testing_set = feature_set[int(len(feature_set)*4/5):]

# creating a naive bayes classifier
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
accuracy = nltk.classify.accuracy(NB_classifier, testing_set)*100
print("Naive Bayes Classifier accuracy =", accuracy)
NB_classifier.show_most_informative_features(20)