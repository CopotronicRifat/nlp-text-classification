def bag_of_words(words):
    return dict([(word, True) for word in words]) 

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words)-set(badwords))

from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)

import collections
def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats

def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats

from nltk.corpus import movie_reviews,brown,reuters
lfeats = label_feats_from_corpus(reuters)

lfeats.keys()

train_feats, test_feats = split_label_feats(lfeats, split=0.75)
print('Number of training features is: ',len(train_feats))
print('Number of testing features is: ',len(test_feats))

from nltk.classify import NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier.train(train_feats)
print('Naive Bayes Classifier labels are: ', nb_classifier.labels())

from nltk.classify.util import accuracy
print('The accuracy of Naive Bayes Classifier is: ', accuracy(nb_classifier, test_feats))
print('Most informative features are: ',nb_classifier.show_most_informative_features(n=15))


from nltk.probability import LaplaceProbDist
nb_classifier = NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
MNB = SklearnClassifier(MultinomialNB())
MNB.train(train_feats)

from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import collections, itertools
import nltk.classify.util, nltk.metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.classify import ClassifierI
from statistics import mode
import sklearn
import collections
from nltk import metrics

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers =classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)

        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf

    
def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    precisions = {}
    recalls = {}
    for label in classifier.labels():
        precisions[label] = metrics.precision(refsets[label], testsets[label])
        recalls[label] = metrics.recall(refsets[label], testsets[label])
    return precisions, recalls

'''
print('\nNAIVE BAYES CLASSIFIER MODELS\n')
nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
print('Positive Precision of Naive Bayes classifier is: ',nb_precisions['religion'])
print('Negetive Precision of Naive Bayes classifier is: ',nb_precisions['news'])
print('Positive Recall of Naive Bayes classifier is: ',nb_recalls['religion'])
print('Negetive Recall of Naive Bayes classifier is: ',nb_recalls['news'])

import nltk
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_feats)
print("\nMultinomial Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(MNB_classifier,test_feats))*100)
mnb_precisions, mnb_recalls = precision_recall(MNB_classifier, test_feats)
print('Positive Precision of Multinomial Naive Bayes classifier is: ',mnb_precisions['religion'])
print('Negetive Precision of Multinomial Naive Bayes classifier is: ',mnb_precisions['news'])
print('Positive Recall of Multinomial Naive Bayes classifier is: ',mnb_recalls['religion'])
print('Negetive Recall of Multinomial Naive Bayes classifier is: ',mnb_recalls['news'])

BernoulliNB_classifier=SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(train_feats)
print("\nBernuolli Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(BernoulliNB_classifier,test_feats))*100)

bnb_precisions, bnb_recalls = precision_recall(BernoulliNB_classifier, test_feats)
print('Positive Precision of Bernuolli Naive Bayes classifier is: ',bnb_precisions['religion'])
print('Negetive Precision of Bernuolli Naive Bayes classifier is: ',bnb_precisions['news'])
print('Positive Recall of Bernuolli Naive Bayes classifier is: ',bnb_recalls['religion'])
print('Negetive Recall of Bernuolli Naive Bayes classifier is: ',bnb_recalls['news'])


print('\nLINEAR CLASSIFIER MODELS')
LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_feats)
print("\nLogistic Regression Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(LogisticRegression_classifier,test_feats))*100)
log_precisions, log_recalls = precision_recall(LogisticRegression_classifier, test_feats)
print('Positive Precision of Logistic Regression classifier is: ',log_precisions['religion'])
print('Negetive Precision of Logistic Regression classifier is: ',log_precisions['news'])
print('Positive Recall of Naive Logistic Regression is: ',log_recalls['religion'])
print('Negetive Recall of Naive Logistic Regression is: ',log_recalls['news'])

SGDClassifier=SklearnClassifier(SGDClassifier())
SGDClassifier.train(train_feats)
print("\nSGD Classifier Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(SGDClassifier,test_feats))*100)
sgd_precisions, sgd_recalls = precision_recall(SGDClassifier, test_feats)
print('Positive Precision of SGD Classifier classifier is: ',sgd_precisions['religion'])
print('Negetive Precision of SGD Classifier classifier is: ',sgd_precisions['news'])
print('Positive Recall of Naive SGD Classifier is: ',sgd_recalls['religion'])
print('Negetive Recall of Naive SGD Classifier is: ',sgd_recalls['news'])

      
print('\nSVM CLASSIFIER MODELS')
LinearSVC=SklearnClassifier(LinearSVC())
LinearSVC.train(train_feats)
print("\nLinear SVC Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(LinearSVC,test_feats))*100)
lsvc_precisions, lsvc_recalls = precision_recall(LinearSVC, test_feats)
print('Positive Precision of LinearSVC classifier is: ',lsvc_precisions['religion'])
print('Negetive Precision of LinearSVC classifier is: ',lsvc_precisions['news'])
print('Positive Recall of LinearSVC classifier is: ',lsvc_recalls['religion'])
print('Negetive Recall of LinearSVC classifier is: ',lsvc_recalls['news'])

SVC_Classifier=SklearnClassifier(SVC())
SVC_Classifier.train(train_feats)
print("\nSVC classifier Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(SVC_Classifier,test_feats))*100)
svc_precisions, svc_recalls = precision_recall(SVC_Classifier, test_feats)
print('Positive Precision of SVC_Classifier classifier is: ',svc_precisions['religion'])
print('Negetive Precision of SVC_Classifier classifier is: ',svc_precisions['news'])
print('Positive Recall of SVC_Classifier classifier is: ',svc_recalls['religion'])
print('Negetive Recall of SVC_Classifier classifier is: ',svc_recalls['news'])

voted_classifier=VoteClassifier(MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier,LinearSVC)
print("\nVoted Classifier Accuracy for Text Classification: ",(nltk.classify.accuracy(voted_classifier,test_feats))*100)


'''

