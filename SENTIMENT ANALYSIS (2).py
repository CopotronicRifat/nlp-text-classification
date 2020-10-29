import nltk
import random
from nltk.corpus import movie_reviews
import collections, itertools
import nltk.classify.util, nltk.metrics

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

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
print('accuracy:', (nltk.classify.util.accuracy(classifier, testing_set))*100)
print('subject precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print('subject recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print('sub precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print('obj recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
