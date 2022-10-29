import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords

###NLTK classifiers work with featstructs. These can be as simple as dictionaries mapping
### feature names to a value. We do this with a default value of true for each word
def bag_of_words(words):
    return dict([(word, True) for word in words])

#### trains a naivebaiyes classifier with the nltk corpus movie reviews data set that consists of
#### positive and negative sentiment evaluations. The classifier will be able to do the same for other sets.
def NaiveBaiyesClassifier():
    negativeIds = movie_reviews.fileids('neg')
    positiveIds = movie_reviews.fileids('pos')

    negfeats = [(bag_of_words(movie_reviews.words(fileids=[f])), 'neg') for f in negativeIds]
    posfeats = [(bag_of_words(movie_reviews.words(fileids=[f])), 'pos') for f in positiveIds]
    trainfeats = negfeats + posfeats
    classifier = NaiveBayesClassifier.train(trainfeats)
    return classifier

#### Open the csv and read the data currently written on the file
data = pd.read_csv('Reddit-Titles.csv')
data_sentences = data['Full_sentence']
data_tickers = data['mention_tickers']

#### create the test_set for the classifier
ndata_sentences = data_sentences[0:10]
test_feats = []
for sentence in ndata_sentences:
    words = sentence.split() 
    test_feats.append((bag_of_words(words),'pos'))


print('accuracy:', nltk.classify.util.accuracy(NaiveBaiyesClassifier(), test_feats))
