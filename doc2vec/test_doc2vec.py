import gensim
from gensim.models import Doc2Vec
from dataset.QCRI import CQRI
import collections
import numpy as np
import os
import re # to remove URL's from tweets
import datetime,time
import string
import nltk
from nltk.stem import PorterStemmer
#from nltk.tokenize.moses import MosesDetokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sacremoses import MosesTokenizer, MosesDetokenizer
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
#For stemming
porter = PorterStemmer()
#Detokenization
detokenizer=MosesDetokenizer()

stop_words = set(stopwords.words('english'))


def clean_single_text(text, date):
    if type(text) == str and type(date) == datetime.datetime:  # because sometimes text is just nothing, so of type None
        text = re.sub(r"http\S+", "", text)  # remove URL's
        text = re.sub(r"@\S+", "", text)  # Optional: remove user names
        text = re.sub(r"\\xa0\S+", "", text)
        text = p.clean(text)
        tokens = word_tokenize(text)
        text = [porter.stem(word) for word in tokens]
        text = [word for word in text if word.isalpha()]

        text = [word.lower() for word in text if word.lower() not in stop_words]

        #text = detokenizer.detokenize(text, return_str=True)
        #text = text.lower()
        return text
    else:
        return None


dataset = CQRI('../twitter.txt')
events = dataset.get_dict()


## SPLIT DATASET IN TRAINING AND TESTING ##
sorted_events_list = sorted(events.items(), key=lambda kv: kv[1])  # sort based on the key, to be able to split the dataset in a deterministic way
training_events_list = sorted_events_list[0:round(0.8*len(sorted_events_list))]
testing_events_list = sorted_events_list[round(0.8*len(sorted_events_list))+1:]
np.save('training_events_list.npy',training_events_list,allow_pickle=True)
np.save('testing_events_list.npy',testing_events_list,allow_pickle=True)
events_training = collections.OrderedDict(training_events_list)
events_testing = collections.OrderedDict(testing_events_list)


#Generating documents
documents_train = []
documents_val = []
counter = 0
for key,val in events_training.items():

    counter += 1
    print(counter)
    if os.path.isfile('../dataset/rumdect/tweets/' + key + '.json'):

        dico = dataset.get_tweets('../dataset/rumdect/tweets/' + key + '.json')

        templist = []

        for tweet_key, tweet_val in dico.items():
                text = clean_single_text(tweet_val[1], tweet_val[0])

                if text is not None:
                    templist.extend(text)

    documents_train.append(gensim.models.doc2vec.LabeledSentence(templist, key))


counter = 0
for key,val in events_testing.items():

    counter += 1
    print(counter)
    if os.path.isfile('../dataset/rumdect/tweets/' + key + '.json'):

        dico = dataset.get_tweets('../dataset/rumdect/tweets/' + key + '.json')

        templist = []

        for tweet_key, tweet_val in dico.items():
                text = clean_single_text(tweet_val[1], tweet_val[0])

                if text is not None:
                    templist.extend(text)

    documents_val.append(gensim.models.doc2vec.LabeledSentence(templist, key))



filename='embedding.d2v'
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
    text_model = Doc2Vec(vector_size=50, min_count=2, epochs=100)
    text_model.build_vocab(documents_train)
    text_model.train(documents_train, total_examples=text_model.corpus_count, epochs=text_model.epochs)
    text_model.save(filename)



train = np.zeros((len(documents_train), 50))
train_labels = np.zeros(len(documents_train))
val = np.zeros((len(documents_val), 50))
val_labels = np.zeros(len(documents_val))

idx = 0
for doc in documents_train:
    vec = text_model.infer_vector(doc.words)
    train[idx,]=vec
    train_labels[idx] = events[doc.tags][1]
    idx += 1

idx = 0
for doc in documents_val:
    vec = text_model.infer_vector(doc.words)
    val[idx,]=vec
    val_labels[idx] = events[doc.tags][1]
    idx += 1


np.save('train_x', train)
np.save('train_y', train_labels)
np.save('val_x', val)
np.save('val_y', val_labels)

