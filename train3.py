import keras
import numpy as np
import utils.utils as u
import sklearn.decomposition as skd
import scipy.io as sio

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
            'epochs':200
            }

    if kwargs is not None:
        for key,val in kwargs.items():
            if key in opts.keys():
                opts[key] = val
            else:
                raise ValueError('Invalid option: '+key)



    #### DATA ####

    train = np.load(train_path)
    test = np.load(test_path)

    #Getting all feature vectors

    inputs_train, labels_train = u.format_inputs_notime(train)
    #inputs_train = np.squeeze(inputs_train, axis=-1)

    inputs_test, labels_test = u.format_inputs_notime(test)
    #inputs_test = np.squeeze(inputs_test, axis=-1)

    # Data normalization
    train_x_mean = np.mean(inputs_train, axis=0)
    train_x_std = np.std(inputs_train, axis=0)

    inputs_train = (inputs_train - train_x_mean)/train_x_std
    inputs_test = (inputs_test - train_x_mean)/train_x_std


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



    #### MODEL ####

    def get_model(embedding_size=32, dropout_rate=0.5, hidden_layers=1, regularization=0.1):
        model = keras.Sequential()

        for i in range(hidden_layers):
            model.add(keras.layers.Dense(embedding_size, kernel_regularizer=keras.regularizers.l2(regularization)))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(rate=dropout_rate))



        model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.1)))
        model.add(keras.layers.Activation('softmax'))

        return model



    ### TRAINING ###

    training_scores = []
    validation_scores = []
    test_scores = []


    for fold in range(foldings):

        temp = np.copy(inputs_train)
        temp_labels = np.copy(labels_train)
        cut_indices = random_indices[cuts[fold]:cuts[fold+1]]

        cut_val_x = np.copy(temp[cut_indices, ])
        cut_train_x = np.delete(temp, cut_indices, 0)

        cut_val_y = np.copy(temp_labels[cut_indices,])
        cut_train_y = np.delete(temp_labels, cut_indices, 0)


        model = get_model(embedding_size=opts['embedding_size'],
                          hidden_layers=opts['hidden_layers'],
                          dropout_rate=opts['dropout_rate'],
                          regularization=opts['regularization'])

        optimizer = keras.optimizers.RMSprop(lr=opts['lr'])

        model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy])


        checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/temp.hdf5', monitor='val_categorical_accuracy', save_best_only=True)
        tb = keras.callbacks.TensorBoard('logs/' + opts['name']+ '/fold_'+str(fold))
        # lrshedule = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_lr=5e-7, verbose=1)


        #### Train ####
        model.fit(x=cut_train_x, y=cut_train_y, validation_data=(cut_val_x, cut_val_y), batch_size=opts['batch_size'],
                  epochs=opts['epochs'], verbose=2, callbacks=[tb, checkpoint], shuffle=True)

        model.load_weights('checkpoints/temp.hdf5')
        training_scores.append(model.evaluate(cut_train_x, cut_train_y)[1])
        validation_scores.append(model.evaluate(cut_val_x, cut_val_y)[1])
        test_scores.append(model.evaluate(inputs_test, labels_test)[1])


    best_fold = np.argmax(validation_scores)

    return(training_scores[best_fold], validation_scores[best_fold], test_scores[best_fold])



if __name__ == '__main__':

    train_path = 'feature_extraction/output_doc2vec_ann/featuresTensor_train_2500.npy'
    test_path = 'feature_extraction/output_doc2vec_ann/featuresTensor_test_2500.npy'

    NAME = 'test_ANN_doc2vec_hiddenlayers_20'
    LR = 1e-4
    EMBEDDING = 20
    REGULARIZATION = 0.1
    DROPOUT = 0.5

    '''val = train_k_fold(train_path, test_path, name=NAME, epochs=500, lr=LR, embedding_size=EMBEDDING,
                       hidden_layers=2, regularization=REGULARIZATION, dropout_rate=DROPOUT)
    print(val)'''


    layers = [1, 2, 3]

    train_score, val_score, test_score = [], [], []

    idx = 0
    for d in layers:
        val = train_k_fold(train_path, test_path, name=NAME+str(idx), epochs=300, lr=LR, embedding_size=EMBEDDING, hidden_layers=d, regularization=REGULARIZATION)
        print(val)

        train_score.append(val[0])
        val_score.append(val[1])
        test_score.append(val[2])

        idx += 1

    print(train_score)
    print(val_score)
    print(test_score)

    dict = {'layers':np.array(layers), 'train':np.array(train_score), 'val':np.array(val_score), 'test':np.array(test_score)}

    sio.savemat('Results/ANN_DOC2VEC/Hiddenlayers_20.mat', dict)


