import keras
import numpy as np
import utils.utils as u
import sklearn.decomposition as skd
import scipy.io as sio
from model import rnn

def train_k_fold(train_path, test_path, **kwargs):

    ### KWARGS ###

    opts = {'foldings':5,
            'embedding_size':32,
            'hidden_layers':1,
            'regularization':0.1,
            'dropout_rate':0.5,
            'lr':1e-3,
            'name':'myNN',
            'batch_size':32,
            'epochs':200,
            'model_type':None,
            'embedding_layer':True
            }

    if kwargs is not None:
        for key,val in kwargs.items():
            if key in opts.keys():
                opts[key] = val
            else:
                raise ValueError('Invalid option: '+key)



    #### DATA ####

    N = 12
    K = 2500

    train = np.load(train_path)
    test = np.load(test_path)


    # Extracting inputs (ndarray of shape (samples, timesteps, nfeatures)
    inputs_train = u.format_inputs(train[:, 0], N, K, None)
    inputs_test= u.format_inputs(test[:, 0], N, K, None)

    # Extracting labels (ndarray of size (samples, 1))
    labels_train = np.array(train[:, 1])
    labels_test = np.array(test[:, 1])


    # Data normalization
    train_x_mean = np.mean(inputs_train, axis=(0,1))
    train_x_std = np.std(inputs_train, axis=(0,1))

    #inputs_train = (inputs_train - train_x_mean)/train_x_std
    #inputs_test = (inputs_test - train_x_mean)/train_x_std

    '''print(np.mean(inputs_test, axis=(0,1)), np.std(inputs_test, axis=(0,1)))
    print(np.mean(inputs_train, axis=(0, 1)), np.std(inputs_train, axis=(0, 1)))'''


    print(len(np.where(labels_train==1)[0]))
    print(len(np.where(labels_train==0)[0]))
    print(len(np.where(labels_test==1)[0]))
    print(len(np.where(labels_test==0)[0]))

    labels_train = keras.utils.to_categorical(labels_train, num_classes=2)
    labels_test = keras.utils.to_categorical(labels_test, num_classes=2)

    # NOTE : THE INPUT DATA SHOULD BE NORMALIZED SOMEHOW


    #### K-FOLD VALIDATION ####

    foldings = opts['foldings']

    #Random shuffle of training data
    random_indices = np.arange(inputs_train.shape[0])
    np.random.shuffle(random_indices)
    cuts = np.floor(np.linspace(0, inputs_train.shape[0], foldings+1)).astype(np.uint16)



    ### TRAINING ###

    training_scores = []
    validation_scores = []
    test_scores = []

    predictions = []


    for fold in range(foldings):

        temp = np.copy(inputs_train)
        temp_labels = np.copy(labels_train)
        cut_indices = random_indices[cuts[fold]:cuts[fold+1]]

        cut_val_x = np.copy(temp[cut_indices, ])
        cut_train_x = np.delete(temp, cut_indices, 0)

        cut_val_y = np.copy(temp_labels[cut_indices,])
        cut_train_y = np.delete(temp_labels, cut_indices, 0)

        model_type = opts['model_type']
        model = model_type(embedding_size=opts['embedding_size'],
                          hidden_layers=opts['hidden_layers'],
                          dropout_rate=opts['dropout_rate'],
                          regularization=opts['regularization'],
                          embedding_layer=opts['embedding_layer'])
        model = model.get_model()

        optimizer = keras.optimizers.RMSprop(lr=opts['lr'])

        model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy])


        #Model checkpoint
        checkpoint_name = 'checkpoints/temp'+str(fold)+'.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_categorical_accuracy', save_best_only=True)

        #Monitoring and early stopping
        tb = keras.callbacks.TensorBoard('logs/' + opts['name']+ '/fold_'+str(fold))
        # lrshedule = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_lr=5e-7, verbose=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

        #### Train ####
        model.fit(x=cut_train_x, y=cut_train_y, validation_data=(cut_val_x, cut_val_y), batch_size=opts['batch_size'],
                   epochs=opts['epochs'], verbose=2, callbacks=[tb, checkpoint, es], shuffle=True)


        #evaluate
        #model.load_weights(checkpoint_name)
        del model

        model = keras.models.load_model(checkpoint_name)
        training_scores.append(model.evaluate(cut_train_x, cut_train_y)[1])
        validation_scores.append(model.evaluate(cut_val_x, cut_val_y)[1])
        test_scores.append(model.evaluate(inputs_test, labels_test)[1])

        predictions.append(model.predict(inputs_test))


    best_fold = np.argmax(validation_scores)

    test_acc = u.majority_voting(np.array(predictions), labels_test)

    outputs = {'best_fold':best_fold,
               'training_scores':training_scores,
               'validation_scores':validation_scores,
               'test_scores':test_scores,
               'predictions':predictions,
               'accuracy':test_acc}


    return(outputs)



if __name__ == '__main__':

    train_path = 'feature_extraction/output_rnn_constant/featuresTensor_train.npy'
    test_path = 'feature_extraction/output_rnn_constant/featuresTensor_test.npy'

    NAME = 'test_GRU2_doc2vec_variable_ensemble'
    LR = 5e-5
    EMBEDDING = 32
    EMBEDDING_LAYER = True
    REGULARIZATION = 0.005
    DROPOUT = 0.5
    MODEL = rnn.GRURNN

    val = train_k_fold(train_path, test_path, name=NAME, epochs=200, lr=LR, embedding_size=EMBEDDING,
                   hidden_layers=2, regularization=REGULARIZATION, dropout_rate=DROPOUT, model_type=MODEL, embedding_layer=EMBEDDING_LAYER)
    print(val['training_scores'])
    print(val['validation_scores'])
    print(val['test_scores'])
    print(val['best_fold'])
    print(val['accuracy'])

    embedding = [20, 40, 80]

    exit()

    train_score, val_score, test_score = [], [], []

    idx = 0
    for d in embedding:
        val = train_k_fold(train_path, test_path, name=NAME+str(idx), epochs=200, lr=LR, embedding_size=d, hidden_layers=1, regularization=REGULARIZATION, dropout_rate=DROPOUT, model_type=MODEL)
        print(val)

        train_score.append(val[0])
        val_score.append(val[1])
        test_score.append(val[2])

        idx += 1

    print(train_score)
    print(val_score)
    print(test_score)

    dict = {'embedding':np.array(embedding), 'train':np.array(train_score), 'val':np.array(val_score), 'test':np.array(test_score)}

    sio.savemat('Results/RNN/GRU2_Embedding.mat', dict)


