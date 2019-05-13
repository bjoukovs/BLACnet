##### IMPORT LIBRAIRIES #####
#Append path
import sys
sys.path.append('../dataset')
#sys.path.append('../tweet_cleaning')
sys.path.append('..')

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



######### LOAD DATASET ##########
# Extract events ID
dataset = CQRI('../twitter.txt')
events = dataset.get_dict()   # start Jupyter with the command line: --NotebookApp.iopub_data_rate_limit=10000000000
                              # for ex.: ipython3 notebook --NotebookApp.iopub_data_rate_limit=10000000000
#print(events)


######### TRAIN TF-IDF ON ALL TEXTS ###########
# Training on the whole tweets dataset to learn the vocabulary
nbrEvents = 0
S_list_total = [];
p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
print("Number of events=",len(events))
counter = 0
#For stemming
porter = PorterStemmer()
#Detokenization
detokenizer=MosesDetokenizer()
for keyEvent in events:
    if os.path.isfile('events/'+keyEvent+'.json'):  # check that the event file exists
        nbrEvents += 1  
        dico = dataset.get_tweets('events/'+keyEvent+'.json')
        # Extraction of the tweet strings and date
        for keyTweet in dico:           #iterates over the keys
            date,text = dico[keyTweet]
            #print(date)

            if type(text) == str and type(date) == datetime.datetime:  # because sometimes text is just nothing, so of type None
                text = re.sub(r"http\S+", "", text) # remove URL's
                text = re.sub(r"@\S+","",text)     # Optional: remove user names
                text=re.sub(r"\\xa0\S+","",text)
                text = p.clean(text)
                tokens = word_tokenize(text)
                text = [porter.stem(word) for word in tokens]
                text = [word for word in text if word.isalpha()]
                text=detokenizer.detokenize(text,return_str=True)
                text=text.lower()
                #print(text)
                S_list_total.append(text)
                
    counter += 1
    print(counter)
            

#np.save('Non_cleaned_tweets.npy',S_list_total)  


#S_list_total=np.load('Non_cleaned_tweets.npy')

vectorizer = TfidfVectorizer()
dummy = vectorizer.fit(S_list_total)
print(vectorizer.vocabulary_)



######## FOR EACH EVENT, CUT IN CONTINUOUS INTERVALS AND APPLY TF-IDF ON EACH INTERVAL  #######
# Parameters
N=12 #reference number of intervals
K = 5000

featuresTensor = [] # list containing tuples (matrixOfFeatures,label), where matrixOfFeatures is a matrix of size K x (number of time interval)

counter = 0

for keyEvent in events:
    counter += 1
    print(counter)
    if os.path.isfile('events/'+keyEvent+'.json'):  # check that the event file exists
        dico = dataset.get_tweets('events/'+keyEvent+'.json')
        ev = events[keyEvent]
        label = ev[1]

        S_list = []
        date_list = []

        # Extraction of the tweet strings and date
        for keyTweet in dico:           #iterates over the keys
            date,text = dico[keyTweet]
            #print(date)

            if type(text) == str and type(date) == datetime.datetime:  # because sometimes text is just nothing, so of type None
                text = re.sub(r"http\S+", "", text) # remove URL's
                text = re.sub(r"@\S+","",text)     # Optional: remove user names
                text=re.sub(r"\\xa0\S+","",text)
                text = p.clean(text)
                tokens = word_tokenize(text)
                text = [porter.stem(word) for word in tokens]
                text = [word for word in text if word.isalpha()]
                text=detokenizer.detokenize(text,return_str=True)
                text=text.lower()
                if text != '':
                    S_list.append(text)
                    date = time.mktime(date.timetuple()) # number of seconds since 1 January 1970
                    date_list.append(date)

        # Sorting w.r.t the date values 
        idx = np.argsort(date_list)
        date_list=np.array(date_list)
        date_list=date_list[idx] # final list is a numpy array sorted in ascending order
        S_list_sorted=[0]*len(S_list)
        for i in range(0,len(S_list)):
            S_list_sorted[i]=S_list[idx[i]]  # S_list is sorted in ascending order w.r.t the date values

        # Time interval
        timeStart = date_list[0]
        timeEnd = date_list[-1]
        totalTimeInterval = timeEnd - timeStart
        timeStep = totalTimeInterval / N  # in [seconds], timeStep is the lowercase 'l' in the paper
        timeStep_init = timeStep # save it for the cut of the intervals

        if timeStep > totalTimeInterval:
            print("timeStep > totalTimeInterval")

        #print(S_list_sorted)

        # Dividing in intervals of duration equal to timeStep, and finding the continuous super-interval, i.e. the longest serie of intervals without empty space
        count_save = 0
        while(True):
            # Dividing in intervals
            S_list_intervals = []  # list of sublists containing tweets belonging to the same interval (each sublist is one interval)
            n = 0 # global counter for the time steps
            i = 0 # counter within an interval
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


            #S_list_intervalsR = [x for x in S_list_intervals if x!=[]]

            #print(S_list_intervals)

            #max_interval = get_max_interval(S_list_intervals)


            # Contiuous interval computation
            continuous_intervals=[] # list of sublists of subsublists: 
                                    # each sublist is a continuous super-interval (continuous = no empty space between intervals)
                                    # each subsublist is one interval inside the continuous super-interval
            count_list=[] # list of the length of the continuous super-intervals
            temp=[]
            count=0
            for elem in S_list_intervals: 
                if(elem==[]):
                    if(temp!=[]):
                        continuous_intervals.append(temp)
                        count_list.append(count)
                    temp=[]
                    count=0
                else:
                    temp.append(elem)
                    count+=1
                    if(elem==S_list_intervals[-1]):
                        continuous_intervals.append(temp)
                        count_list.append(count)

            count_list=np.array(count_list)
            idx_max = np.argmax(count_list)
            max_interval = continuous_intervals[idx_max]  # super-interval covering the longest time span
            count_max = count_list[idx_max]               # number of intervals (and so time steps) in max_interval 
            print("Count_max = ",count_max)

            if (count_max < N and count_max > count_save): # Half the time step and restart at the beginning of while(True)
                #print('Redo')
                timeStep = timeStep/2 # shorten the time interval by doubling N
                count_save = count_max
                max_interval_save = max_interval 
            else:                                         # Output max_interval and count_max
                #print('Done')
                if timeStep != timeStep_init:
                    max_interval = max_interval_save # when outputting take the previous iteration result, that was the best because current iteration didn't improve
                    print("Final count = ", count_save)
                    print("Final interval content:\n")
                    #print(max_interval_save)
                else:
                    print("Final count = ", count_max)
                    print("Final interval content:\n")
                    #print(max_interval)

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
        featuresMat = np.zeros((K,len(max_interval)))
        #print(max_interval)
        for ii in range(0,len(max_interval)):
            #featuresMat[:,ii] = tf_idf(max_interval[ii],True,K)   # to modify to take each interval separately
            tmp = vectorizer.transform(max_interval[ii])
            array = tmp.toarray()
            #print(array.shape)
            vec = np.reshape(array,(array.shape[0]*array.shape[1],1))
            vec = np.ndarray.flatten(vec)
            vec[::-1].sort()
            vec = vec[0:K]
            #print(vec)
            #print(vec.shape) 
            #vec = array[np.nonzero(array)]
            #vec[::-1].sort()
            #vec = np.transpose(vec[0:K])
            featuresMat[:,ii] = vec
        #print(featuresMat)

        featuresTensor.append((featuresMat,label))

        #break # TO TEST ONLY ONE EVENT
#print(featuresTensor)

np.save('featuresTensor.npy',featuresTensor)
#featuresTensor=np.load('featuresTensor.npy')