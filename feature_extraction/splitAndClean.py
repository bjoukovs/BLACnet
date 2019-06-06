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
    ######## PART 1: getting dataset, splitting it and cleaning it #######
    print('Main1')

    ## LOAD DATASET ##
    # Extract events ID
    dataset = CQRI('../twitter.txt')
    events = dataset.get_dict()

    print(events)

    ## SPLIT DATASET IN TRAINING AND TESTING ##
    sorted_events_list = sorted(events.items(), key=lambda kv: kv[1])  # sort based on the key, to be able to split the dataset in a deterministic way
    random.shuffle(sorted_events_list)
    training_events_list = sorted_events_list[0:round(0.85*len(sorted_events_list))]
    testing_events_list = sorted_events_list[round(0.85*len(sorted_events_list))+1:]
    np.save('training_events_list.npy',training_events_list,allow_pickle=True)
    np.save('testing_events_list.npy',testing_events_list,allow_pickle=True)
    events_training = collections.OrderedDict(training_events_list)
    events_testing = collections.OrderedDict(testing_events_list)

    # Training on the whole tweets dataset to learn the vocabulary
    nbrEvents = 0
    S_list_total = []
    S_list_total_val = []
    print("Number of events=",len(events))


    #(Training + Validation) set
    S_list_total = functions.clean_set_eventIsDoc(events_training,dataset)
    np.save('cleaned_tweets/cleaned_tweets_train.npy',S_list_total)
    #Testing set
    S_list_total_val = functions.clean_set_eventIsDoc(events_testing,dataset)
    np.save('cleaned_tweets/cleaned_tweets_test.npy',S_list_total_val)