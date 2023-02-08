import matplotlib.pyplot as plt
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords

## NLTK classifiers work with featstructs. These can be as simple as dictionaries mapping
## feature names to a value. We do this with a default value of true for each word
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
    #print('accuracy:', nltk.classify.util.accuracy(classifier, trainfeats))
    return classifier

def Eval_text(title):
        classifier = NaiveBaiyesClassifier()
        sentiment = classifier.classify({'l': title}) 
        return sentiment
