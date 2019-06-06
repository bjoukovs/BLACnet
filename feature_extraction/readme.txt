Note: all functions for this part are saved in functions.py, and there are 3 scripts to launch: splitAndClean.py, extractFeatures_tfidf.py and extracFeatures_doc2vec.py

Required librairies:
- preprocessor
- sklearn
- numpy
- scipy
- nltk: once nltk is installed, you need to run the command nltk.download() and install all the components of nltk
- sacremoses
- gensim

1) Run splitAndClean.py to split the dataset into training, validation and testing and to clean the tweets. You can change the directories to which you save files.

2) TF-IDF: run extractFeatures_tfidf.py: there are several options for the feature extraction: variable interval for RNN, fixed intervals for RNN, and one block approach for ANN. Again you can change the directories to which you save files.

3) Doc2vec: run extracFeatures_doc2vec.py: this is done for the ANN approach only. Again you can change the directories to which you save files.