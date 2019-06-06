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
    if len(inputs_train.shape) == 3: inputs_train = np.squeeze(inputs_train, axis=-1)

    inputs_test, labels_test = u.format_inputs_notime(test)
    if len(inputs_test.shape) == 3: inputs_test = np.squeeze(inputs_test, axis=-1)

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



        model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(regularization)))
        model.add(keras.layers.Activation('softmax'))

        return model



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


        model = get_model(embedding_size=opts['embedding_size'],
                          hidden_layers=opts['hidden_layers'],
                          dropout_rate=opts['dropout_rate'],
                          regularization=opts['regularization'])

        optimizer = keras.optimizers.RMSprop(lr=opts['lr'])

        # Model checkpoint
        checkpoint_name = 'checkpoints/temp' + str(fold) + '.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_categorical_accuracy',
                                                     save_best_only=True)

        model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy])


        tb = keras.callbacks.TensorBoard('logs/' + opts['name']+ '/fold_'+str(fold))
        # lrshedule = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_lr=5e-7, verbose=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)


        #### Train ####
        model.fit(x=cut_train_x, y=cut_train_y, validation_data=(cut_val_x, cut_val_y), batch_size=opts['batch_size'],
                  epochs=opts['epochs'], verbose=2, callbacks=[tb, checkpoint, es], shuffle=True)


        #Evaluation
        del model
        model = keras.models.load_model(checkpoint_name)
        training_scores.append(model.evaluate(cut_train_x, cut_train_y)[1])
        validation_scores.append(model.evaluate(cut_val_x, cut_val_y)[1])
        test_scores.append(model.evaluate(inputs_test, labels_test)[1])

        predictions.append(model.predict(inputs_test))

    best_fold = np.argmax(validation_scores)

    test_acc = u.majority_voting(np.array(predictions), labels_test)

    outputs = {'best_fold': best_fold,
               'training_scores': training_scores,
               'validation_scores': validation_scores,
               'test_scores': test_scores,
               'predictions': predictions,
               'accuracy': test_acc}

    return (outputs)



if __name__ == '__main__':

    train_path = 'feature_extraction/output2/featuresTensor_train.npy'
    test_path = 'feature_extraction/output2/featuresTensor_val.npy'

    NAME = 'ANN_TFIDF_FINAL_1layer'
    LR = 1e-4
    EMBEDDING = 40
    REGULARIZATION = 0.1
    DROPOUT = 0.5

    val = train_k_fold(train_path, test_path, name=NAME, epochs=300, lr=LR, embedding_size=EMBEDDING,
                       hidden_layers=1, regularization=REGULARIZATION, dropout_rate=DROPOUT)
    print(val['accuracy'])
    print(np.mean(np.array(val['validation_scores'])))
    print(np.std(np.array(val['validation_scores'])))

    exit()


    '''reg = [0.2, 0.5, 0.65]

    train_score, val_score, test_score, accuracies = [], [], [], []

    idx = 0
    for d in reg:
        val = train_k_fold(train_path, test_path, name=NAME+str(idx), epochs=300, lr=LR, embedding_size=EMBEDDING, hidden_layers=2, regularization=REGULARIZATION, dropout_rate=d)
        print(val)

        train_score.append(val['training_scores'])
        val_score.append(val['validation_scores'])
        test_score.append(val['test_scores'])
        accuracies.append(val['accuracy'])

        idx += 1

    print(accuracies)
    dict = {'dropout':np.array(reg), 'train':np.array(train_score), 'val':np.array(val_score), 'test':np.array(test_score), 'accuracy':np.array(accuracies)}

    sio.savemat('Results/ANN/FINAL_DOC2VEC.mat', dict)'''


