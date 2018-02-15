import nltk
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
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer as keras_token

# fix random seed for reproducibility
np.random.seed(7)

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
filtered = list(filter(lambda x: x[6] == 1.0 and x[5] != "unknown", filtered))

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
    word_list = " ".join(list(map(lambda x: try_str(x[19]), lst)))
    #word_list = word_list.lower().split(" ")
    word_list = wordify(word_list).split(" ")
    word_list = list(w for w in word_list if len(w) > 1 and w not in stop_words)
    return word_list


def word_distrebution(text, messeg, f):
    ret_list = []
    list_words = " ".join(list(map(f, text)))
    #list_words = " ".join(list(map(lambda x: try_str(x[19]), text)))
    #list_words = list_words.lower().split(" ")
    list_words = wordify(list_words).split(" ")
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



male_features = word_distrebution(males, 'Male most frequent words', lambda x: try_str(x[10])+try_str(x[19]))

print ()
female_features = word_distrebution(females, 'Female most frequent words', lambda x: try_str(x[10])+try_str(x[19]))

print ()
brand_features = word_distrebution(brands, 'Brand most frequent words', lambda x: try_str(x[10])+try_str(x[19]))

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
#
# # creating a naive bayes classifier
# NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
# accuracy = nltk.classify.accuracy(NB_classifier, testing_set)*100
# print("Naive Bayes Classifier accuracy =", accuracy)
# NB_classifier.show_most_informative_features(20)
#
# # creating a logistic regression classifier
# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
# accuracy = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100
# print("Logistic Regression classifier accuracy =", accuracy)

# creating a nural network classifier
neural_data_set = list(filter(lambda x: x[1] != "brand", tweet_by_gender))
x = list(neural_data_set[i][0] for i in range (0, len(neural_data_set)))
encoder = LabelEncoder()
y = encoder.fit_transform(list(neural_data_set[i][1] for i in range(0, len(neural_data_set))))

#x = x[:500]
#y = y[:500]


max_words = 4000
#max_words = 40
k_tokenizer = keras_token(num_words=max_words)
k_tokenizer.fit_on_texts(x)
x = k_tokenizer.texts_to_sequences(x)
x = sequence.pad_sequences(x)
# treat the labels as categories
y = keras.utils.to_categorical(y, 2)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)


embed_dim = 128
lstm_out = 196

model = load_model('my_model.h5')
# model = Sequential()
#
# model.add(Embedding(max_words, embed_dim,input_length = x.shape[1]))
# model.add(LSTM(lstm_out))
# model.add(Dense(2,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=256)

#4 epoches, batch 256 - acc - 63.3
#8 epoches,batch 256 - acc - 66.76
#8 epoches, batch 500 - acc - 63.83
#8 epoches, batch 100 - acc -64.96

# Final evaluation of the model
validation_size = 500
X_validate = X_test[-validation_size:]
Y_validate = y_test[-validation_size:]
X_test = X_test[:-validation_size]
y_test = y_test[:-validation_size]

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#model.save('my_model.h5')
# q 4  - testint the model on data that was collected in real time


raw_tweets = pd.read_csv('realTime.csv', encoding='utf8')
list_tweets =  list(raw_tweets.values[i][0] for i in range (0, len(raw_tweets.values)))

#
to_predict = k_tokenizer.texts_to_sequences(list_tweets)
to_predict = sequence.pad_sequences(to_predict, maxlen=54L)


predictions = model.predict(to_predict, batch_size = 200)
#round predictions were 0 is female
                     #  1 is male
rounded = [round(x[0]) for x in predictions]
num_of_men = rounded.count(1.0)
num_of_women = len(rounded) - num_of_men

print('women percentage in collected tweets',)
print(num_of_women)






