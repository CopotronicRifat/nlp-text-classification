import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import collections, itertools
import nltk.classify.util, nltk.metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
import random
from nltk.classify import ClassifierI
from statistics import mode
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
#from classification import precision_recall


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
    

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

word_features = list(all_words.keys())[:3000]


def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1
    n_xx = label_word_fd.N()
    high_info_words = set()
    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)
        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
    return high_info_words

def bag_of_words_in_set(words, goodwords):
    return bag_of_words(set(words) & set(goodwords))

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

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
        precisions[label] = nltk.metrics.precision(refsets[label], testsets[label])
        recalls[label] = nltk.metrics.recall(refsets[label], testsets[label])
    return precisions, recalls


labels = movie_reviews.categories()
labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

#NAIVE BAYES CLASSIFIER
classifier=nltk.NaiveBayesClassifier.train(training_set)
#classifier_f=open("naivebayes.pickle","rb")
#classifier=pickle.load(classifier_f)
#classifier_f.close()
print("Original Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(classifier,testing_set))*100)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
nb_precisions, nb_recalls = precision_recall(classifier, testing_set)
print('nb_precision is:',nb_precisions['pos'])
#print('precision: ', nltk.metrics.precision(classifier, testing_set))
print('nb_precision is:',nb_precisions['neg'])
print('nb_recall is:',nb_recalls['pos'])
print('nb_recall is:',nb_recalls['neg'])
classifier.show_most_informative_features(15)

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
mnb_precisions, mnb_recalls = precision_recall(MNB_classifier, testing_set)
print('mnb_precision is:',mnb_precisions['pos'])
print('mnb_precision is:',mnb_precisions['neg'])
print('mnb_recall is:',mnb_recalls['pos'])
print('mnb_recall is:',mnb_recalls['neg'])


random.shuffle(featuresets)
training_set=featuresets[0:100]
testing_set=featuresets[100:300]
#GaussianNB_classifier=SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("Gaussian Naive Bayes Algorithm Accuracy for Lambda Calculas Lexical Semantic Parsing: ",(nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)

BernoulliNB_classifier=SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("Bernuolli Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

random.shuffle(featuresets)
training_set=featuresets[0:100]
testing_set=featuresets[100:300]
LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Logistic Regression Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

random.shuffle(featuresets)
training_set=featuresets[0:100]
testing_set=featuresets[100:300]
SGDClassifier=SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGD Classifier Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(SGDClassifier,testing_set))*100)

random.shuffle(featuresets)
training_set=featuresets[0:100]
testing_set=featuresets[100:300]
SVC_Classifier=SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC classifier Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(SVC_Classifier,testing_set))*100)

random.shuffle(featuresets)
training_set=featuresets[0:100]
testing_set=featuresets[100:300]
LinearSVC=SklearnClassifier(LinearSVC())
LinearSVC.train(training_set)
print("Linear SVC Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(LinearSVC,testing_set))*100)

#random.shuffle(featuresets)
#training_set=featuresets[0:100]
#testing_set=featuresets[100:300]
#NuSVC_Classifier=SklearnClassifier(NuSVC(nu=0.5))
#NuSVC_Classifier.train(training_set)
#print("NuSVC Algorithm Accuracy for Lambda Calculas Lexical Semantic Parsing: ",(nltk.classify.accuracy(NuSVC_Classifier,testing_set))*100)


#print('accuracy:', (nltk.metrics.precision(LinearSVC, testing_set))*100)
voted_classifier=VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier,LinearSVC)
print("\nVoted Classifier Accuracy for Text Classification: ",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
#print("Classification: ",voted_classifier.classify(testing_set[0][0]),"Confidence% :",voted_classifier.confidence(testing_set[0][0])*100)
#print("Classification: ",voted_classifier.classify(testing_set[1][0]),"Confidence% :",voted_classifier.confidence(testing_set[1][0])*100)


#SAVE CLASSIFIER WITH PICKLE
#save_classifier=open("naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()


