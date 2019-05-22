##### IMPORT LIBRAIRIES #####
#Append path
import sys
sys.path.append('../dataset')
#sys.path.append('../tweet_cleaning')
sys.path.append('..')
sys.path.append('output/')

# Import OS
import os

# Librairies for preprocessing the tweets
#import io
#import unittest

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


##TRAIN TF-IDF ON ALL TEXTS: DO IT ONE TIME AND THEN JUST LOAD FILE ##

p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
#For stemming
porter = PorterStemmer()
#Detokenization
detokenizer=MosesDetokenizer()


def clean_single_text(text, date):
    if type(text) == str and type(date) == datetime.datetime:  # because sometimes text is just nothing, so of type None
        text = re.sub(r"http\S+", "", text)  # remove URL's
        text = re.sub(r"@\S+", "", text)  # Optional: remove user names
        text = re.sub(r"\\xa0\S+", "", text)
        text = p.clean(text)
        tokens = word_tokenize(text)
        text = [porter.stem(word) for word in tokens]
        text = [word for word in text if word.isalpha()]
        text = detokenizer.detokenize(text, return_str=True)
        text = text.lower()
        return text
    else:
        return None



def clean_event(keyEvent):

    output = None

    if os.path.isfile('../dataset/rumdect/tweets/'+keyEvent+'.json'):  # check that the event file exists

        output = []

        dico = dataset.get_tweets('../dataset/rumdect/tweets/'+keyEvent+'.json')
        # Extraction of the tweet strings and date
        for keyTweet in dico:           #iterates over the keys
            date,text = dico[keyTweet]
            #print(date)

            text = clean_single_text(text, date)

            if text is not None:
                output.append(text)


    return output


def clean_set_tweetIsDoc(events):
    counter = 0
    total_list = []
    for keyEvent in events:

        partial_list = clean_event(keyEvent)
        if partial_list is not None:
            total_list.extend(partial_list)

        counter += 1
        print(counter)

    return total_list

def clean_set_eventIsDoc(events):
    separator = ' '
    total_list = []
    counter = 0
    for keyEvent in events:

        partial_list = clean_event(keyEvent)
        if partial_list is not None:
            partial_list = separator.join(partial_list)
            total_list.append(partial_list)

        counter += 1
        print(counter)
    return total_list


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
                    tmp = vectorizer.transform([interval])
                    vec = tmp.toarray()
                    #print(vec)
                    #vec = np.append(vec,np.zeros((1,K-vec.shape[1])))
                    featuresMat[:, ii] = vec
                # print(featuresMat)

                featuresTensor.append((featuresMat, label))

                #break # TO TEST ONLY ONE EVENT

    return featuresTensor


def extractFeatures(dataset, events, vectorizer, K=5000):
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
            vector = vectorizer.transform([full_text]).toarray()

            featuresMatrix.append((vector, label))

    return featuresMatrix


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

                if len(S_list_intervals) != N:
                    print("ERROR: the number of intervals should be equal to N")
                    exit()

                # Apply tf_idf on each interval of the super-interval
                featuresMat = np.zeros((K, N))
                # print(max_interval)
                for ii in range(0,N):
                    # featuresMat[:,ii] = tf_idf(max_interval[ii],True,K)   # to modify to take each interval separately
                    separator = ' '
                    interval = separator.join(max_interval[ii])
                    tmp = vectorizer.transform([interval])
                    vec = tmp.toarray()
                    #print(vec)
                    #vec = np.append(vec,np.zeros((1,K-vec.shape[1])))
                    featuresMat[:, ii] = vec
                # print(featuresMat)

                featuresTensor.append((featuresMat, label))

                #break # TO TEST ONLY ONE EVENT

    return featuresTensor





######## PART 1: getting dataset, splitting it and cleaning it #######

## LOAD DATASET ##
# Extract events ID
dataset = CQRI('../twitter.txt')
events = dataset.get_dict()   # start Jupyter with the command line: --NotebookApp.iopub_data_rate_limit=10000000000
                              # for ex.: ipython3 notebook --NotebookApp.iopub_data_rate_limit=10000000000

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
S_list_total = clean_set_eventIsDoc(events_training)
np.save('cleaned_tweets_train.npy',S_list_total)
#Testing set
S_list_total_val = clean_set_eventIsDoc(events_testing)
np.save('cleaned_tweets_test.npy',S_list_total_val)


######## END OF PART 1: TO COMMENT WHEN S_list_total IS SAVED #######


"""
####### PART 2: CUT IN INTERVAL AND EXTRACT FEATURES #######

# Parameters
N = 12 #reference number of intervals
K = 2500

#Train vectorizer
S_list_total=np.load('output/cleaned_tweets_train.npy')
vectorizer = TfidfVectorizer(max_features=K,stop_words='english')
dummy = vectorizer.fit(S_list_total)
#print(vectorizer.vocabulary_)
#del S_list_total # delete this variable to free memory


#Extract features

training_events_list = np.load('output/training_events_list.npy',allow_pickle=True) # load training event list
events_training = collections.OrderedDict(training_events_list) # convert it back to dictionary

val_events_list = np.load('output/testing_events_list.npy',allow_pickle=True) # load training event list
events_val = collections.OrderedDict(val_events_list) # convert it back to dictionary


dataset = CQRI('../twitter.txt') # recreate it here when first part is commented

#featuresTensor = cut_intervals_extract_features(dataset=dataset, events=events_training, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
featuresTensor = extractFeatures(dataset=dataset, events=events_training, vectorizer=vectorizer, K=K)
np.save('output2/featuresTensor_train.npy',featuresTensor)

#featuresTensor = cut_intervals_extract_features(dataset=dataset, events=events_val, vectorizer=vectorizer, N=N, K=K) # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)
featuresTensor = extractFeatures(dataset=dataset, events=events_val, vectorizer=vectorizer, K=K)
np.save('output2/featuresTensor_test.npy',featuresTensor)


#print(featuresTensor)


#featuresTensor=np.load('featuresTensor.npy')

######## END OF PART 2 ########
"""