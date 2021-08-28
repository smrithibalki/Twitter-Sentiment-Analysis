#!/usr/bin/env python
# coding: utf-8

# In[5]:


import string
from bs4 import BeautifulSoup
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sys import stdout
import itertools
import pandas
import re

folder_path = ''
train_file_name = "training-Obama-Romney-tweets.xlsx"
test_file_name_1 = "final-testData-no-label-Obama-tweets.xlsx"
test_file_name_2="final-testData-no-label-Romney-tweets.xlsx"
process_test_file = True

stemmer = SnowballStemmer('english')

stop_words = stopwords.words('english')
stop_words.remove('not')

exclude = set(string.punctuation)
patterns_to_replace = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'),
                       (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                       (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'),
                       (r'(\w+)\'d', '\g<1> would')]
patterns = [(re.compile(regex), repl) for (regex, repl) in patterns_to_replace]


def ExpandContractions(tweet):
    text = tweet
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


def DelimitBySpace(tweet):
    ''' Example - ThisisASentence --> This is A Sentence'''
    return re.sub("([a-z])([A-Z])", "\g<1> \g<2>", tweet)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' smile ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' laugh ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' love ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' wink ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' sad ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' cry ', tweet)
    return tweet


def preprocess(tweet):
    # tweet = str(tweet)
    tweet = tweet.lower()

    # remove html tags
    tweet = BeautifulSoup(tweet, 'html.parser').get_text()

    # remove @ references
    tweet = re.sub(r'@\w+', ' ', tweet)

    # remove hashtags
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove 'RT' text
    tweet = re.sub(r'(^| )rt ', ' ', tweet)

    # remove links
    tweet = re.sub(r'https?:\/\/\S*', ' ', tweet)

    # Removing URls
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))', '', tweet)

    # Voteeeeeeeee -> Votee
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    # handle emojis
    tweet = handle_emojis(tweet)

    # Replace the hex code "\xe2\x80\x99" with single quote
    tweet = re.sub(r'\\xe2\\x80\\x99', "'", tweet)

    # Removing apostrophe
    tweet = tweet.replace("\'s", '')

    # Expanding contractions. For example, "can't" will be replaced with "cannot"
    tweet = ExpandContractions(tweet)

    # Removing punctuation
    tweet = ''.join(ch for ch in tweet if ch not in exclude)

    # Removing words that end with digits
    tweet = re.sub(r'\d+', '', tweet)

    # Removing words that start with a number or a special character
    tweet = re.sub(r"^[^a-zA-Z]+", ' ', tweet)

    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

    # Replace all words that don't start with a letter, number or an underscore with an empty string
    tweet = re.sub(r'\\[xa-z0-9.*]+', '', tweet)

    # Remove trailing spaces and full stops
    tweet = tweet.strip(' .')

    # Convert words such as ThisIsAsentence --> This is A Sentence
    tweet = DelimitBySpace(tweet)

    # tokenize
    words = word_tokenize(tweet)

    # remove stopwords
    words = [stemmer.stem(w) for w in words if not w in stop_words]

    # join the words
    tweet = " ".join(words)

    return tweet


data = []


def cleanTweets_store_in_File(tweets, file_name):
    if not process_test_file:
        # remove class 2 data
        tweets = tweets.loc[tweets.iloc[:, 1].isin([1, -1, 0])]

    tweets = tweets.astype(str)
    for i, row in tweets.iterrows():
        Tweet_cleaned = preprocess(row[0])
        if process_test_file:
              label=''
        else:
            label = str(row[1])
        tweet_tuple = Tweet_cleaned, label
        # print(tweet_tuple)
        data.append(tweet_tuple)

    if process_test_file:
        '''If test_file is cleaned store it in file 
            with extension "_test_cleaned.csv"
        '''
        data_frame = pandas.DataFrame(data, columns=['tweet', 'label'])
        data_frame.to_csv(folder_path+ file_name + "_test_cleaned.csv")
    else:
        '''If train_file is cleaned store it in file 
             with extension "_train_cleaned.csv"
        '''
        data_frame = pandas.DataFrame(data, columns=['tweet', 'label'])
        data_frame.to_csv(folder_path+file_name + "_train_cleaned.csv")

    stdout.write("\b" * 50 + "Cleaning %s file...  " % (file_name))
    data.clear()
    print("")


#for candidate in ['Obama', 'Romney']:
candidate_1="Obama"
candidate_2="Romney"
if process_test_file:
    tweets_1 = pandas.read_excel(folder_path + test_file_name_1, sheet_name="Obama", usecols='B')
    tweets_2 = pandas.read_excel(folder_path + test_file_name_2, sheet_name="Romney", usecols='B')
else:
    tweets = pandas.read_excel(folder_path + train_file_name, sheet_name="Obama", usecols='D:E')
cleanTweets_store_in_File(tweets_1, candidate_1.lower())
cleanTweets_store_in_File(tweets_2, candidate_2.lower())


# In[ ]:




