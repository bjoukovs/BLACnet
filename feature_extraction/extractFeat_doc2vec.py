##### IMPORT LIBRAIRIES #####
#Append path
import sys
sys.path.append('../dataset')
#sys.path.append('../tweet_cleaning')
sys.path.append('..')
sys.path.append('output/')

# Import OS
import os

# CQRI to get tweets
from QCRI import CQRI
import preprocessor as p

# Librairies for computations and ML
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix

# Python utility tools
import re # to remove URL's from tweets
import datetime,time
import string
import nltk
from nltk.stem import PorterStemmer
#from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize import word_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer

import collections # to sort the dictionary
import operator
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from extractFeat import *


# Main part
# Parameters
N = 12 #reference number of intervals
K = 2500
train_doc2vec(K) # do it once and then comment this line
model= Doc2Vec.load("d2v.model")


def extractFeatures_doc2vec(dataset, events, vectorizer, K=5000):
    '''
    For the simple ANN model
    '''
    counter = 0
    featuresMatrix = []

    for keyEvent in events:
        counter += 1
        print(counter)
        if os.path.isfile('../dataset/rumdect/tweets/' + keyEvent + '.json'):  # check that the event file exists
            dico = dataset.get_tweets('../dataset/rumdect/tweets/' + keyEvent + '.json')
            ev = events[keyEvent]
            label = ev[1]

            S_list = []

            for keyTweet in dico:  # iterates over the keys
                date, text = dico[keyTweet]
                # print(date)

                text = clean_single_text(text, date)

                if text is not None:
                    S_list.append(text)


            full_text = ' '.join(S_list)

            test_data = word_tokenize(full_text.lower())
            vector = model.infer_vector(test_data)
            featuresMatrix.append((vector, label))

    return featuresMatrix


def train_doc2vec(K):
    max_epochs = 100
    vec_size = K
    alpha = 0.025

    S_list_total=np.load('output/cleaned_tweets_train.npy') # each event is a document
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")







#Extract features

training_events_list = np.load('output/training_events_list.npy',allow_pickle=True) # load training event list
events_training = collections.OrderedDict(training_events_list) # convert it back to dictionary

val_events_list = np.load('output/testing_events_list.npy',allow_pickle=True) # load training event list
events_val = collections.OrderedDict(val_events_list) # convert it back to dictionary


dataset = CQRI('../twitter.txt') # recreate it here when first part is commented

#featuresTensor = cut_intervals_extract_features(dataset=dataset, events=events_training, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
featuresTensor =
np.save('output2/featuresTensor_train.npy',featuresTensor)

#featuresTensor = cut_intervals_extract_features(dataset=dataset, events=events_val, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
featuresTensor =
np.save('output2/featuresTensor_test.npy',featuresTensor)