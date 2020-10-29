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

from nltk.classify import ClassifierI
from statistics import mode


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
    


sub_term=open("WORDNET/sub.txt","r").read()
obj_term=open("WORDNET/obj.txt","r").read()

docs=[]
for r in sub_term.split(' '):
    docs.append((r,"sub"))

for r in obj_term.split(' '):
    docs.append((r,"obj"))




#STOP WORD SEPARATION
stop_words=set(stopwords.words("english"))

allwords=[]
sub_words=word_tokenize(sub_term)
obj_words=word_tokenize(obj_term)


for w in sub_words:
    allwords.append(w.lower())

for w in obj_words:
    allwords.append(w.lower())
allwords=nltk.FreqDist(allwords)
word_features=list(allwords.keys())[:300]

#WORDNET DATASET ARCHIVE
from nltk.corpus import wordnet
from nltk.corpus import PlaintextCorpusReader
corpus_root = 'E:\EIGHTH SEMESTER\PROJECT AND THESIS II\SOFTWARE\WORDNET'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
training_data=wordlists.sents('document.txt')


#INPUT TEXTS
sentence = input("Please enter the text: ")
print("The input string: ", sentence)
corpora=sentence
wordtoken_test=word_tokenize(sentence)
wordtoken_train=training_data[0]
print("Tokenization of training data: ",wordtoken_train[0:10])
print("Tokenization of testing data: ",wordtoken_test)


#PARTS OF SPEECH TAGGING OF WORDS
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
post_train=nltk.pos_tag(wordtoken_train)
post_test=nltk.pos_tag(wordtoken_test)
print("Parts of speech tagging of training data:",post_train[0:10])
print("Parts of speech tagging of testing data:",post_test)



#PRINT THE TEXTS
featured_word=[]
connectives=[]
subject_part=[]
verb_part=[]
object_part=[]
pos_tagging=[]
connectives2=[]
subject_part2=[]
verb_part2=[]
object_part2=[]
pos_tagging2=[]


#CATEGORIZE THE TRAINING DATA
for i in range(0,len(wordtoken_train)):
    
    if 'CNJ' in post_train[i][1]:
        connectives.append(post_train[i][1])
    elif 'VBD' in post_train[i][1]:
        verb_part.append(post_train[i][1])
    elif 'V' in post_train[i][1]:
        verb_part.append(post_train[i][1])
    elif 'VG' in post_train[i][1]:
        verb_part.append(post_train[i][1])
    elif 'VD' in post_train[i][1]:
        verb_part.append(post_train[i][1])
    elif 'VN' in post_train[i][1]:
        verb_part.append(post_train[i][1])
    elif 'PRP' in post_train[i][1]:
        subject_part.append(post_train[i][1])
        pos_tagging.append('SUB')
    elif 'NP' in post_train[i][1]:
        subject_part.append(post_train[i][1])
        pos_tagging.append('SUB')
    elif 'N' in post_train[i][1]:
        subject_part.append(post_train[i][1])
        pos_tagging.append('SUB')
    else:
        object_part.append(post_train[i][1])
        pos_tagging.append('OBJ')
    

#CATEGORIZE THE TESTING DATA
for i in range(0,len(wordtoken_test)):
    if 'CNJ' in post_test[i][1]:
        connectives2.append(post_test[i][1])
    elif 'VBD' in post_test[i][1]:
        verb_part2.append(post_test[i][1])
    elif 'V' in post_test[i][1]:
        verb_part2.append(post_test[i][1])
    elif 'VG' in post_test[i][1]:
        verb_part2.append(post_test[i][1])
    elif 'VD' in post_test[i][1]:
        verb_part2.append(post_test[i][1])
    elif 'VN' in post_test[i][1]:
        verb_part2.append(post_test[i][1])
    elif 'PRP' in post_test[i][1]:
        subject_part2.append(post_test[i][1])
        pos_tagging2.append('SUB')
    elif 'NP' in post_test[i][1]:
        subject_part2.append(post_test[i][1])
        pos_tagging2.append('SUB')
    elif 'N' in post_test[i][1]:
        subject_part2.append(post_test[i][1])
        pos_tagging2.append('SUB')
    else:
        object_part2.append(post_test[i][1])
        pos_tagging2.append('OBJ')


#PARTS OF SPEECH TAGSET
#ADJ adjective new, good, high, special, big, local
#ADV adverb really, already, still, early, now
#CNJ conjunction and, or, but, if, while, although
#DET determiner the, a, some, most, every, no
#EX existential there, there’s
#FW foreign word dolce, ersatz, esprit, quo, maitre
#MOD modal verb will, can, would, may, must, should
#N noun year, home, costs, time, education
#NP proper noun Alison, Africa, April, Washington
#NUM number twenty-four, fourth, 1991, 14:24
#PRO pronoun he, their, her, its, my, I, us
#P preposition on, of, at, with, by, into, under
#TO the word to to
#UH interjection ah, bang, ha, whee, hmpf, oops
#V verb is, has, get, do, make, see, run
#VD past tense said, took, told, made, asked
#VG present participle making, going, playing, working
#VN past participle given, taken, begun, sung
#WH wh determiner who, which, when, what, where, how



#NAMED ENTITY RECOGNITION
NameEntity=nltk.ne_chunk(post_test)
NameEntity.draw()
category_sub=['SUB']
category_obj=['OBJ']
category=['SUB','OBJ']
#TEXT CLASSIFICATION
import random

#print('training data: ')
#print(post_train)

#print("Connective Part: ",connectives)
#print("Subject Part: ",subject_part)
#print("Verb Part: ",verb_part)
#print("Object Part: ",object_part)

documents=([(wordlists,'sub') for wordlists in subject_part]+
           [(wordlists,'obj') for wordlists in object_part])

documents2=([(corpora,'sub') for corpora in subject_part2]+
           [(corpora,'obj') for corpora in object_part2])

random.shuffle(documents)
random.shuffle(documents2)
#print("Documents: ",documents)

all_words=[]
for w in wordtoken_train:
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)
#word_features=list(all_words.keys())

def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)

    return features

featuresets=[(find_features(w),c) for (w, c) in docs]
featuresets2=[(find_features(w2),c2) for (w2, c2) in documents2]

random.shuffle(featuresets)

#TRAINING AND TESTING SET INITIALIZATION
training_set=featuresets[0:100]
testing_set=featuresets[100:300]

#NAIVE BAYES CLASSIFIER
classifier=nltk.NaiveBayesClassifier.train(training_set)
#classifier_f=open("naivebayes.pickle","rb")
#classifier=pickle.load(classifier_f)
#classifier_f.close()
print("Original Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(10)

MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algorithm Accuracy for Text Classification: ",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

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
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

#print('accuracy:', (nltk.metrics.precision(LinearSVC, testing_set))*100)
print('subject precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print('subject recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print('sub precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print('obj recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
voted_classifier=VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifier,LinearSVC)
print("\nVoted Classifier Accuracy for Text Classification: ",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
#print("Classification: ",voted_classifier.classify(testing_set[0][0]),"Confidence% :",voted_classifier.confidence(testing_set[0][0])*100)
#print("Classification: ",voted_classifier.classify(testing_set[1][0]),"Confidence% :",voted_classifier.confidence(testing_set[1][0])*100)


#SAVE CLASSIFIER WITH PICKLE
#save_classifier=open("naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()


