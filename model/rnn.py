import keras

class BasicRNN():
    def __init__(self, embedding_size=100, **kwargs):

        reg = kwargs['regularization']
        drp = kwargs['dropout_rate']

        self.model = keras.Sequential()

        self.model.add(keras.layers.BatchNormalization())


        #Embedding
        self.model.add(keras.layers.Dense(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg)))

        # RNN
        self.model.add(keras.layers.SimpleRNN(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Dropout(rate=drp))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class GRURNN():
    def __init__(self, embedding_layer=True, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        reg = kwargs['regularization']
        drp = kwargs['dropout_rate']

        self.model.add(keras.layers.BatchNormalization())

        #input layer
        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(reg)))
            #self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.GRU(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Dropout(rate=drp))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class LSTMRNN():
    def __init__(self, embedding_layer=True, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        reg = kwargs['regularization']
        drp = kwargs['dropout_rate']

        self.model.add(keras.layers.BatchNormalization())

        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(reg)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.LSTM(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Dropout(rate=drp))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model


class LSTM2RNN():
    def __init__(self, embedding_layer=True, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        reg = kwargs['regularization']
        drp = kwargs['dropout_rate']

        self.model.add(keras.layers.BatchNormalization())

        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(reg)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.LSTM(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg), return_sequences=True))
        self.model.add(keras.layers.Dropout(rate=drp))
        self.model.add(keras.layers.LSTM(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Dropout(rate=drp))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class GRU2RNN():
    def __init__(self, embedding_layer=True, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        reg = kwargs['regularization']
        drp = kwargs['dropout_rate']

        self.model.add(keras.layers.BatchNormalization())

        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(reg)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.GRU(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg), return_sequences=True))
        self.model.add(keras.layers.Dropout(rate=drp))
        self.model.add(keras.layers.GRU(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(reg), return_sequences=False))
        self.model.add(keras.layers.Dropout(rate=drp))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(reg)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model
