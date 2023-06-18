from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = stopwords.words('english')
'''
This dataset contains:
- 50.000 negative sentiments
- 50.000 positive sentiments
- 50.000 no sentiments
'''

# Tokenizing the data
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

# Normalizing the data
'''
We need to convert different forms of a word to a normalised one.
E.g. run, ran, rans => to run

In this process we will use stemming and lemmatization.
Stemming is the process of removing affixes from a word.
Lemmatization will normalize the words with the context of the vocabulary and morphological analysis of word in text.
'''

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []

    for word, tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n' # token is a noun
        elif tag.startswith("VB"):
            pos = 'v' # token is a verb
        else:
            pos = 'a'
        
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    
    return lemmatized_sentence

# Removing noise from data
'''
Noise is specific for each project. Since we are working with tweets we will focus on this items:
- Hyperlinks: Hyperlinks in twitter are converted to the URL shortner `t.co`
- Twitter handles in replies: Usernames are preceded by the symbol `@` which doesn't add any meanning
- Punctuation and special characters: Even though they add meaning to a text it is more harder to process
so we will remove them.
'''

import re, string

def remove_noise(tokens, stop_words = {}):
    cleaned_tokens = []

    for token, tag in pos_tag(tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    
    return cleaned_tokens

positive_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tweets = []
negative_cleaned_tweets = []

for token in positive_tweets_tokens:
    positive_cleaned_tweets.append(remove_noise(lemmatize_sentence(token), stop_words=stop_words))

for token in negative_tweets_tokens:
    negative_cleaned_tweets.append(remove_noise(lemmatize_sentence(token), stop_words=stop_words))

def get_all_tokens(cleaned_tokens):
    for tokens in cleaned_tokens:
        for token in tokens:
            yield token

all_pos_words = get_all_tokens(positive_cleaned_tweets)
from nltk import FreqDist

freq_dist = FreqDist(all_pos_words)

# Preparing data for the model
'''
Sentiment analysis is the attitude of the author to the topic that it is written about.
'''

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tweets)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tweets)

import random
import pickle

positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy", classify.accuracy(classifier, test_data))
pickle.dump(classifier, open("Model.sav", "wb"))
