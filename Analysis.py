#The os library is for interacting with the operating system.
import os
#The nltk library is used to work with human language data.
import nltk
#The random library is useful for random numbers.
import random
#The pickle library implements an algorithm to transform an arbitrary Python object into a series of bytes.
import pickle
#This module provides functions for the calculation of mathematical statistics of numerical data.
from statistics import mode
#word_tokenize separates words from each other through appropriate separators such as space or comma.
from nltk.tokenize import word_tokenize
#Stopwords are empty, meaningless words.
from nltk.corpus import stopwords
#The re library is for regular expressions.
import re

# Opening files with IA training sets.
files_pos = os.listdir('train/pos')
files_pos = [open('train/pos/' + f, 'r').read() for f in files_pos]
files_neg = os.listdir('train/neg')
files_neg = [open('train/neg/' + f, 'r').read() for f in files_neg]

# Variable for the list containing all words in the structured text following
# the NLP pipeline. It will then contain the frequency of the words in the text.
all_words = []

# Variable for the list of both positive and negative tuples.
documents = []

# List to contain stopwords.
stop_words = list(set(stopwords.words('english')))

# To reduce computational complexity we only focus on adjectives in the reviews.
# J stands for adjective
allowed_word_types = ["J"]

# For each file in the positive reviews list
for p in files_pos:
    # Create a list of tuples where the first element of each tuple is a review
    # the second element is the label which is, in this case, "pos" because
    # we're handling positive reviews; each tuple is added to the list documents.
    documents.append((p, "pos"))

    # Remove punctuations
    # (a-zA-Z) matches every letter
    # \s matches Unicode whitespace characters
    # r means the string will be treated as raw string so \s
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # Tokenize the sentence without punctuation.
    tokenized = word_tokenize(cleaned)

    # Remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # Parts of speech tagging for each word.
    pos = nltk.pos_tag(stopped)

    # Make a list of  all adjectives identified by the allowed word types list above.
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# For each file in the negative reviews list we repeat the same operations as before
for p in files_neg:
    # Create a list of tuples where the first element of each tuple is a review
    # the second element is the label which is, in this case, "neg" because
    # we're handling positive reviews; each tuple is added to the list documents.
    documents.append((p, "neg"))

    # Remove punctuations
    # (a-zA-Z) matches every letter
    # \s matches Unicode whitespace characters
    # r means the string will be treated as raw string so \s
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # Tokenize the sentence without punctuation
    tokenized = word_tokenize(cleaned)

    # Remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # Parts of speech tagging for each word.
    neg = nltk.pos_tag(stopped)

    # Make a list of all adjectives identified by the allowed word types list above.
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# Creating a frequency distribution of each adjectives.
all_words = nltk.FreqDist(all_words)

# Listing the 5000 most frequent words.
word_features = list(all_words.keys())[:5000]


# Function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features.
# The values of each key are either true or false for wether that feature appears in the review or not.

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# Creating features for each review
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents
random.shuffle(featuresets)

training_set = featuresets[:20000]
testing_set = featuresets[20000:]

from nltk.classify import ClassifierI


# Defining the ensemble model class to combine different models together

class EnsembleClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # Returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # A simple measurement of the degree of confidence in the classification
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# Load all classifiers from the pickled files

# Function to load models given filepath using pickle
def load_model(file_path):
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Original Naive Bayes Classifier
ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier
MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')

# Bernoulli Naive Bayes Classifier
BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

# Initializing the ensemble classifier
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf)

# List of only feature dictionary from the featureset list of tuples
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]


# Function that given a review tells us the classification and confidence
def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)


# Tester
text_a = '''Spider-Man: Homecoming is a bad film.'''
ris = sentiment(text_a)
# Creation of the result
if ris[0] == 'pos':
    predizione = "La recensione è positiva al "
else:
    predizione = "La recensione è negativa al "

print(predizione + str(ris[1] * 100))
