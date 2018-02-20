from nltk.probability import FreqDist
import string
import pandas as pd
from nltk.corpus import stopwords

stop_words = stopwords.words('english') + list(string.punctuation)

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





raw_tweets = pd.read_csv('realTime.csv', encoding='utf8')
list_tweets =  list(raw_tweets.values[i][0] for i in range (0, len(raw_tweets.values)))
word_distrebution(list_tweets,'common words in our dataset: ',lambda x:x)