import nltk
import random
from nltk.corpus import movie_reviews, brown, reuters

documents1 = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(documents1)
print(documents1[1])


all_words1 = []
for u in movie_reviews.words():
     all_words1.append(u.lower())


all_words1 = nltk.FreqDist(all_words1)
print(all_words1.most_common(15))
