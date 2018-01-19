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
from keras.models import Sequential
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
        st = str(t)
        st = st.lower()
        return st
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

#
# creating a nural network classifier
neural_data_set = list(filter(lambda x: x[1] != "brand", tweet_by_gender))
x = list(neural_data_set[i][0] for i in range (0, len(neural_data_set)))
encoder = LabelEncoder()
y = encoder.fit_transform(list(neural_data_set[i][1] for i in range(0, len(neural_data_set))))

max_words = 4000

max_text_length = 400
k_tokenizer = keras_token(num_words=max_words)
k_tokenizer.fit_on_texts(x)

x = k_tokenizer.texts_to_sequences(x)
x = sequence.pad_sequences(x)
# treat the labels as categories
y = keras.utils.to_categorical(y, 2)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)



embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_words, embed_dim,input_length = x.shape[1]))
model.add(LSTM(lstm_out))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=256)

# Final evaluation of the model
validation_size = 500
X_validate = X_test[-validation_size:]
Y_validate = y_test[-validation_size:]
X_test = X_test[:-validation_size]
y_test = y_test[:-validation_size]

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")

