import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import unicodecsv as csv


def getLiveData():
    consumer_key = 'uW7b9X2txbXPgXXPjHNxk3yvZ'
    consumer_secret = 'nxQ38FkZkkkf3OSrL1pZwYbRBXRhTeKVetQtpecxtOsW0R7Erz'
    access_token = '33584667-xIa3A136SDuAk32JNqkZZYKqnwUFA2s30hKs8qUAp'
    access_secret = '59nOtlV0fEuS9qeN0DeroMqsWQ45Qmcn0Os5IqmnNWDvd'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

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


getLiveData()
