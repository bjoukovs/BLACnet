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
    print('Main3')

    # Parameters
    N = 12 #reference number of intervals
    K = 2500

    # Train doc2vec once, then comment next line
    #functions.train_doc2vec(K) # do it once and then comment this line
    model= Doc2Vec.load("d2v_2500.model")

    # Load list of events from splitAndClean.py
    training_events_list = np.load('training_events_list.npy',allow_pickle=True) # load training event list
    events_training = collections.OrderedDict(training_events_list) # convert it back to dictionary
    val_events_list = np.load('testing_events_list.npy',allow_pickle=True) # load training event list
    events_val = collections.OrderedDict(val_events_list) # convert it back to dictionary
    dataset = CQRI('../twitter.txt') # recreate it here when first part is commented

    # Extract features for ANN
    featuresTensor = functions.extractFeatures_doc2vec(dataset, events_training, model, K=K)
    np.save('output_doc2vec_ann/featuresTensor_test_2500.npy',featuresTensor)