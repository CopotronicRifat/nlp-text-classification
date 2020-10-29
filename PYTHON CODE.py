
#import packages from NLTK
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

#movie review corpus e 2 ta file ase, ekta positive ekta negetive
#ekhane file id gula ekta variable e nise
def evaluate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
 
    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
#dataset ke vag kora hoise
#kisu portion testing set ar kisu traing set hisebe
    negcutoff = 1000
    poscutoff = 1000

    
# 0 theke negcut index porjonto trainfeature ar tarpor theke testing set 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

#naive bayes claassifer apply kora hoise ekhane trainset er upor
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

#ekhane F1 score metric use kore accuracy measure korse
# google e "F1 score" likehe search dile bujte parbi eta kivabe calculate kore
    print('accuracy: ', nltk.classify.util.accuracy(classifier, testfeats))
    print('pos precision:', nltk.precision(refsets['pos'], testsets['pos']))
    print('pos recall:', nltk.recall(refsets['pos'], testsets['pos']))
    print('neg precision:', nltk.precision(refsets['neg'], testsets['neg']))
    print('neg recall:', nltk.recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features(10)
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
#evalute_classifer function call kora hoyeche (single word er jonno)
#mane ekta word ekta kore feature
print('evaluating single word features')
evaluate_classifier(word_feats)

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()


#ekhane son word ke small latter e niye asha hoyeche
#karon jeno for example "Run" and "run" eki word hisebe count hoy
for word in movie_reviews.words(categories=['pos']):
    word_fd[word.lower()] += 1
    label_word_fd['pos'][word.lower()] += 1
 
for word in movie_reviews.words(categories=['neg']):
    word_fd[word.lower()] += 1
    label_word_fd['neg'][word.lower()] += 1
 
# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()

#individual word gulo count kora hoyese ekhane
#positive, negetive ar total word er count kora hoyeche
pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count


#ekhane bigram method use kora hoise (chi square) metric hisebe nise
#statistics e chilo, amar obosso mone nai
#naive bayes er moto arekta method
word_scores = {}
 
for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],(freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],(freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

#word gulo ke sort kore best 10000 word neya hoise
best = sorted(word_scores.iteritems(), reverse=True)[:10000]
bestwords = set([w for w, s in best])
 
def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])

#then abar evelute_classifier call kora hoyeche
#ekhane best word gulo hocche feature
print('evaluating best word features')
evaluate_classifier(best_word_feats)
 
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d

#abar evelute_classifier fuction call kora hoyeche (bigram+bestword) method apply kore
#bigram mane hocche dual relation between two word
print('evaluating best words + bigram chi_sq word features')
evaluate_classifier(best_bigram_word_feats)
