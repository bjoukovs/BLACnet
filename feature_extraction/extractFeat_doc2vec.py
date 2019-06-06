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

#from gistfile1 import read_npy_chunk,read_npy_chunk_demo_unsafe

from feature_extraction.extractFeat import clean_single_text

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

    S_list_total=np.load('output_rnn_constant/cleaned_tweets_train.npy',mmap_mode='r') # each event is a document
    #S_list_total = read_npy_chunk('cleaned_tweets_train.npy',1,100)
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(S_list_total)]

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        #print('iteration {0}'.format(epoch))
        print("Doc2vec training epoch: ",epoch)
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v_2500.model")
    print("Model Saved")

def cut_intervals_extract_features(dataset, events, vectorizer, N=12, K=5000):
    '''
    For the RNN model with variable size non-empty intervals
    '''
    counter = 0
    featuresTensor = []

    for keyEvent in events:
        counter += 1
        print(counter)
        if os.path.isfile('../dataset/rumdect/tweets/' + keyEvent + '.json'):  # check that the event file exists
            dico = dataset.get_tweets('../dataset/rumdect/tweets/' + keyEvent + '.json')
            ev = events[keyEvent]
            label = ev[1]

            S_list = []
            date_list = []

            # Extraction of the tweet strings and date
            for keyTweet in dico:  # iterates over the keys
                date, text = dico[keyTweet]
                # print(date)

                text = clean_single_text(text, date)

                if text is not None:
                    S_list.append(text)
                    date = time.mktime(date.timetuple())  # number of seconds since 1 January 1970
                    date_list.append(date)

            if date_list != []:  # to avoid the case with empty date list so with only empty tweets
                # Sorting w.r.t the date values
                idx = np.argsort(date_list)
                date_list = np.array(date_list)
                date_list = date_list[idx]  # final list is a numpy array sorted in ascending order
                S_list_sorted = [0] * len(S_list)
                for i in range(0, len(S_list)):
                    S_list_sorted[i] = S_list[idx[i]]  # S_list is sorted in ascending order w.r.t the date values

                #dummy = vectorizer.fit(S_list_sorted) # learn the vocabulary on each event separately

                # Time interval
                timeStart = date_list[0]
                timeEnd = date_list[-1]
                totalTimeInterval = timeEnd - timeStart
                timeStep = totalTimeInterval / N  # in [seconds], timeStep is the lowercase 'l' in the paper
                timeStep_init = timeStep  # save it for the cut of the intervals

                if timeStep > totalTimeInterval:
                    print("timeStep > totalTimeInterval")

                # print(S_list_sorted)

                # Dividing in intervals of duration equal to timeStep, and finding the continuous super-interval, i.e. the longest serie of intervals without empty space
                count_save = 0
                while (True):
                    # Dividing in intervals
                    S_list_intervals = []  # list of sublists containing tweets belonging to the same interval (each sublist is one interval)
                    n = 0  # global counter for the time steps
                    i = 0  # counter within an interval
                    timeUp = 0
                    while timeUp < timeStart + totalTimeInterval:
                        timeDown = timeStart + n * timeStep
                        timeUp = timeDown + timeStep

                        interval = []
                        while i < len(date_list):
                            if date_list[i] <= timeUp and date_list[i] >= timeDown:
                                interval.append(S_list_sorted[i])
                                i += 1
                            else:
                                break
                        S_list_intervals.append(interval)
                        n += 1

                    # S_list_intervalsR = [x for x in S_list_intervals if x!=[]]

                    # print(S_list_intervals)

                    # max_interval = get_max_interval(S_list_intervals)

                    # Contiuous interval computation
                    continuous_intervals = []  # list of sublists of subsublists:
                    # each sublist is a continuous super-interval (continuous = no empty space between intervals)
                    # each subsublist is one interval inside the continuous super-interval
                    count_list = []  # list of the length of the continuous super-intervals
                    temp = []
                    count = 0
                    for elem in S_list_intervals:
                        if (elem == []):
                            if (temp != []):
                                continuous_intervals.append(temp)
                                count_list.append(count)
                            temp = []
                            count = 0
                        else:
                            temp.append(elem)
                            count += 1
                            if (elem == S_list_intervals[-1]):
                                continuous_intervals.append(temp)
                                count_list.append(count)

                    count_list = np.array(count_list)
                    idx_max = np.argmax(count_list)
                    max_interval = continuous_intervals[idx_max]  # super-interval covering the longest time span
                    count_max = count_list[idx_max]  # number of intervals (and so time steps) in max_interval
                    print("Count_max = ", count_max)

                    if (count_max < N and count_max > count_save):  # Half the time step and restart at the beginning of while(True)
                        # print('Redo')
                        timeStep = timeStep / 2  # shorten the time interval by doubling N
                        count_save = count_max
                        max_interval_save = max_interval
                    else:  # Output max_interval and count_max
                        # print('Done')
                        if timeStep != timeStep_init:
                            max_interval = max_interval_save  # when outputting take the previous iteration result, that was the best because current iteration didn't improve
                            print("Final count = ", count_save)
                            print("Final interval content:\n")
                            # print(max_interval_save)
                        else:
                            print("Final count = ", count_max)
                            print("Final interval content:\n")
                            # print(max_interval)

                        break

                # Check lengths
                countLen_maxInterval = 0
                countLen_S_list = 0;
                for elem in max_interval:
                    countLen_maxInterval += len(elem)
                for elem2 in S_list_intervals:
                    if elem2 != []:
                        countLen_S_list += len(elem2)
                if countLen_maxInterval > countLen_S_list:
                    print("Problem: there shouldn't be more tweets in the biggest interval than the total number of tweets")

                # Apply tf_idf on each interval of the super-interval
                featuresMat = np.zeros((K, len(max_interval)))
                # print(max_interval)
                for ii in range(0, len(max_interval)):
                    # featuresMat[:,ii] = tf_idf(max_interval[ii],True,K)   # to modify to take each interval separately
                    separator = ' '
                    interval = separator.join(max_interval[ii])
                    tmp = vectorizer.infer_vector([interval])
                    #print(vec)
                    #vec = np.append(vec,np.zeros((1,K-vec.shape[1])))
                    featuresMat[:, ii] = tmp
                # print(featuresMat)

                featuresTensor.append((featuresMat, label))

                #break # TO TEST ONLY ONE EVENT

    return featuresTensor


def cutSameIntervals_extractFeatures(dataset, events, vectorizer, N=12, K=5000):
    '''
    Separate the events in a FIXED number of intervals, which have the SAME size. There may thus be
    EMPTY intervals.
    '''
    counter = 0
    featuresTensor = []

    for keyEvent in events:
        counter += 1
        print(counter)
        if os.path.isfile('../dataset/rumdect/tweets/' + keyEvent + '.json'):  # check that the event file exists
            dico = dataset.get_tweets('../dataset/rumdect/tweets/' + keyEvent + '.json')
            ev = events[keyEvent]
            label = ev[1]

            S_list = []
            date_list = []

            # Extraction of the tweet strings and date
            for keyTweet in dico:  # iterates over the keys
                date, text = dico[keyTweet]
                # print(date)

                text = clean_single_text(text, date)

                if text is not None:
                    S_list.append(text)
                    date = time.mktime(date.timetuple())  # number of seconds since 1 January 1970
                    date_list.append(date)

            if date_list != []:  # to avoid the case with empty date list so with only empty tweets
                # Sorting w.r.t the date values
                idx = np.argsort(date_list)
                date_list = np.array(date_list)
                date_list = date_list[idx]  # final list is a numpy array sorted in ascending order
                S_list_sorted = [0] * len(S_list)
                for i in range(0, len(S_list)):
                    S_list_sorted[i] = S_list[idx[i]]  # S_list is sorted in ascending order w.r.t the date values

                #dummy = vectorizer.fit(S_list_sorted) # learn the vocabulary on each event separately

                # Time interval
                timeStart = date_list[0]
                timeEnd = date_list[-1]
                totalTimeInterval = timeEnd - timeStart
                timeStep = totalTimeInterval / N  # in [seconds], timeStep is the lowercase 'l' in the paper
                timeStep_init = timeStep  # save it for the cut of the intervals

                if timeStep > totalTimeInterval:
                    print("timeStep > totalTimeInterval")

                # print(S_list_sorted)

                # Dividing in intervals of duration equal to timeStep, and finding the continuous super-interval, i.e. the longest serie of intervals without empty space
                count_save = 0

                # Dividing in intervals
                S_list_intervals = []  # list of sublists containing tweets belonging to the same interval (each sublist is one interval)
                n = 0  # global counter for the time steps
                i = 0  # counter within an interval
                timeUp = 0
                while timeUp < timeStart + totalTimeInterval:
                    timeDown = timeStart + n * timeStep
                    timeUp = timeDown + timeStep

                    interval = []
                    while i < len(date_list):
                        if date_list[i] <= timeUp and date_list[i] >= timeDown:
                            interval.append(S_list_sorted[i])
                            i += 1
                        else:
                            break
                    S_list_intervals.append(interval)
                    n += 1

                if len(S_list_intervals) != N:
                    print("ERROR: the number of intervals should be equal to N")
                    exit()

                # Apply tf_idf on each interval of the super-interval
                featuresMat = np.zeros((K, N))
                # print(max_interval)
                for ii in range(0,N):
                    # featuresMat[:,ii] = tf_idf(max_interval[ii],True,K)   # to modify to take each interval separately
                    separator = ' '
                    interval = separator.join(S_list_intervals[ii])
                    tmp = vectorizer.infer_vector([interval])
                    #print(vec)
                    #vec = np.append(vec,np.zeros((1,K-vec.shape[1])))
                    featuresMat[:, ii] = tmp
                # print(featuresMat)

                featuresTensor.append((featuresMat, label))

                #break # TO TEST ONLY ONE EVENT

    return featuresTensor




# Main part
# Parameters
N = 12 #reference number of intervals
K = 2500
#train_doc2vec(K) # do it once and then comment this line



model= Doc2Vec.load("d2v_2500.model")
#Extract features
training_events_list = np.load('training_events_list.npy',allow_pickle=True) # load training event list
events_training = collections.OrderedDict(training_events_list) # convert it back to dictionary
val_events_list = np.load('testing_events_list.npy',allow_pickle=True) # load training event list
events_val = collections.OrderedDict(val_events_list) # convert it back to dictionary

dataset = CQRI('../twitter.txt') # recreate it here when first part is commented

#featuresTensor = extractFeatures_doc2vec(dataset, events_training, model, K=K)
#featuresTensor = cut_intervals_extract_features(dataset, events_training, vectorizer=model, K=K, N=N)
#featuresTensor = cutSameIntervals_extractFeatures(dataset, events_training, vectorizer=model, K=K, N=N)
#np.save('output_doc2vec_rnn_variable/featuresTensor_train_2500.npy',featuresTensor)
#featuresTensor = extractFeatures_doc2vec(dataset, events_val, model, K=K)
#featuresTensor = cut_intervals_extract_features(dataset, events_val, vectorizer=model, K=K, N=N)
featuresTensor = cutSameIntervals_extractFeatures(dataset, events_val, vectorizer=model, K=K, N=N)
np.save('output_doc2vec_rnn_constant/featuresTensor_test_2500.npy',featuresTensor)



