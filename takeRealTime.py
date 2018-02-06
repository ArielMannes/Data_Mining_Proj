from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import unicodecsv as csv
from nltk.probability import FreqDist
import pandas as pd
from nltk.corpus import stopwords
import string


# Collect live data


def getLiveData():
    consumer_key = 'uW7b9X2txbXPgXXPjHNxk3yvZ'
    consumer_secret = 'nxQ38FkZkkkf3OSrL1pZwYbRBXRhTeKVetQtpecxtOsW0R7Erz'
    access_token = '33584667-xIa3A136SDuAk32JNqkZZYKqnwUFA2s30hKs8qUAp'
    access_secret = '59nOtlV0fEuS9qeN0DeroMqsWQ45Qmcn0Os5IqmnNWDvd'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    class MyListener(StreamListener):
        def on_data(self, data):
            data2 = json.loads(data)
            if not data2['entities']['urls']:
                description = data2['user']['description']
                if not isinstance(description, basestring):
                    description = ""
                text = description + ' ' + data2['text']
                print(text)
                try:
                    with open('realTime.csv', 'a') as f:

                        writer = csv.writer(f, dialect='excel', encoding='utf-8')
                        writer.writerow([text])
                        return True
                except BaseException as e:
                    print("Error on_data: %s" % str(e))
                return True

        def on_error(self, status):
            print(status)
            return True

    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(languages=['en'],
                          track=['nail', 'nails', 'feminism', '#nails', '#feminism', '#female', '#princess',
                                 '#feelings',
                                 '#aprincess', '#fashiondesigner', '#future', '#passion', '#dedication', '#year',
                                 '#milliondollarlisting', '#fetuses', '#china', '#death', '#born', '#policy', '#info',
                                 '#didyouknow', '#instagram', '#insta', '#instagood', '#instafact', '#instapic',
                                 '#like4like', '#likeforlike', '#tag', '#tagsforlikes', '#instatag', '#instame'])


def readData():
    with open('realTime.csv') as f:
        allData = []
        reader = csv.reader(f, encoding='utf-8')
        for row in reader:
            allData.append(row[0])

        return allData


stop_words = stopwords.words('english') + list(string.punctuation) + ['RT', 'rt', '&amp;']


def word_distrebution(text, messeg, f):
    ret_list = []
    list_words = " ".join(list(map(f, text)))
    list_words = list_words.lower().split(" ")
    list_words = list(w for w in list_words if len(w) > 1 and w not in stop_words)
    word_dist = FreqDist(list_words)
    feature_set = word_dist.most_common(4000)
    word_dist = word_dist.most_common(10)
    dist = pd.DataFrame.from_records(word_dist).transpose
    print (messeg)
    print dist

    for word in feature_set:
        ret_list.append(word[0])

    return ret_list


i = readData()

word_distrebution(i, "real time twits", lambda x: x)
print len(i)
