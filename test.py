import keras
import numpy as np
import utils.utils as u
import sklearn.decomposition as skd
import scipy.io as sio

N = None
K = None

############################################################################################

### PLEASE UNCOMMENT YOUR TEST CASE ###

# ANN TF-IDF #
TYPE = 'ann_tfidf'
TEST_PATH = 'feature_extraction/output2/featuresTensor_val.npy'
TRAIN_PATH = 'feature_extraction/output2/featuresTensor_train.npy'
NET_NAME = 'ANN_TFIDF'

'''
# ANN DOC2VEC #
TYPE = 'ann_doc2vec'
TEST_PATH = 'feature_extraction/output_doc2vec_ann/featuresTensor_test_2500.npy'
TRAIN_PATH = 'feature_extraction/output_doc2vec_ann/featuresTensor_train_2500.npy'
NET_NAME = 'ANN_DOC2VEC'
'''
'''
# RNN TF-IDF VARIABLE #
TYPE = 'rnn_tfidf_variable'
TEST_PATH = 'feature_extraction/output_rnn_variable/featuresTensor_test.npy'
TRAIN_PATH = 'feature_extraction/output_rnn_variable/featuresTensor_train.npy'
NET_NAME = 'LSTM2_VARIABLE'
N=12
K=2500
'''
'''
# RNN TF-IDF CONSTANT #
TYPE = 'rnn_tfidf_constant'
TEST_PATH = 'feature_extraction/output_rnn_constant/featuresTensor_test.npy'
TRAIN_PATH = 'feature_extraction/output_rnn_constant/featuresTensor_train.npy'
NET_NAME = 'RNN_CONSTANT'
N=12
K=2500

'''

##############################################################################

def format_imputs(inputs, type, train_path, N=0, K=0):

    if type == 'ann_tfidf' or type == 'ann_doc2vec':
        train = np.load(train_path)
        inputs_train, labels_train = u.format_inputs_notime(train)
        if len(inputs_train.shape) == 3: inputs_train = np.squeeze(inputs_train, axis=-1)

        inputs_test, labels_test = u.format_inputs_notime(inputs)
        if len(inputs_test.shape) == 3: inputs_test = np.squeeze(inputs_test, axis=-1)

        # Data normalization
        train_x_mean = np.mean(inputs_train, axis=0)
        train_x_std = np.std(inputs_train, axis=0)

        inputs_test = (inputs_test - train_x_mean) / train_x_std

        labels_test = keras.utils.to_categorical(labels_test, num_classes=2)

    elif type == 'rnn_tfidf_variable' or type=='rnn_tfidf_constant':

        train = np.load(train_path)
        # Extracting inputs (ndarray of shape (samples, timesteps, nfeatures)
        inputs_train = u.format_inputs(train[:, 0], N, K, None)
        inputs_test = u.format_inputs(inputs[:, 0], N, K, None)

        # Extracting labels (ndarray of size (samples, 1))
        labels_test = np.array(inputs[:, 1])

        labels_test = keras.utils.to_categorical(labels_test, num_classes=2)



    return (inputs_test, labels_test)



if __name__ == '__main__':

    if N is None: N=0
    if K is None: K=0

    test_set = format_imputs(np.load(TEST_PATH), TYPE, TRAIN_PATH, N, K)
    inputs_test = test_set[0]
    labels_test = test_set[1]

    predictions = []
    for fold in range(5):
        print('Predicting fold', fold+1)
        model = keras.models.load_model('checkpoints/'+NET_NAME+str(fold)+'.hdf5')
        predictions.append(model.predict(inputs_test))

    test_acc = u.majority_voting(np.array(predictions), labels_test)
    print('Test set accuracy:', test_acc)