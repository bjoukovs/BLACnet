import functions

##### IMPORT LIBRAIRIES #####
#Append path
import sys
sys.path.append('../dataset')
sys.path.append('..')
sys.path.append('output/')

# Import OS
import os

# CQRI to get tweets
from dataset.QCRI import CQRI
import preprocessor as p

# Librairies for computations and ML
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix

# Python utility tools
import re # to remove URL's from tweets
import datetime,time
import string
import collections # to sort the dictionary
import operator
import random

# Natural language processing kit
import nltk
from nltk.stem import PorterStemmer
#from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize import word_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer

# Gensim for doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

if __name__ == "__main__":
####### PART 2: CUT IN INTERVAL AND EXTRACT FEATURES #######
    print('Main2')

    # Parameters
    N = 12 #reference number of intervals
    K = 2500

    #Train vectorizer
    S_list_total=np.load('cleaned_tweets/cleaned_tweets_train.npy')
    vectorizer = TfidfVectorizer(max_features=K,stop_words='english')
    dummy = vectorizer.fit(S_list_total)

    # Load list of events from splitAndClean.py
    training_events_list = np.load('training_events_list.npy',allow_pickle=True) # load training event list
    events_training = collections.OrderedDict(training_events_list) # convert it back to dictionary
    val_events_list = np.load('testing_events_list.npy',allow_pickle=True) # load training event list
    events_val = collections.OrderedDict(val_events_list) # convert it back to dictionary
    dataset = CQRI('../twitter.txt') # recreate it here when first part is commented


    #### Choices for feature extraction: ####
    # 1) variable intervals for RNN
    featuresTensor = functions.cut_intervals_extract_features(dataset=dataset, events=events_training, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
    #np.save('output_rnn_variable/featuresTensor_train.npy',featuresTensor)
    featuresTensor = functions.cut_intervals_extract_features(dataset=dataset, events=events_val, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
    #np.save('output_rnn_variable/featuresTensor_test.npy',featuresTensor)

    # 2) fixed intervals for RNN
    #featuresTensor = functions.cutSameIntervals_extractFeatures(dataset=dataset, events=events_training, vectorizer=vectorizer, K=K)
    #np.save('output_rnn_fixed/featuresTensor_train.npy',featuresTensor)
    #featuresTensor = functions.cutSameIntervals_extractFeatures(dataset=dataset, events=events_val, vectorizer=vectorizer, K=K)
    #np.save('output_rnn_fixed/featuresTensor_test.npy',featuresTensor)

    # 3) no interval
    #featuresTensor = functions.extractFeatures(dataset=dataset, events=events_training, vectorizer=vectorizer, K=K)
    #np.save('output_ann/featuresTensor_train.npy',featuresTensor)
    #featuresTensor = functions.extractFeatures(dataset=dataset, events=events_val, vectorizer=vectorizer, K=K)
    #np.save('output_ann/featuresTensor_test.npy',featuresTensor)
