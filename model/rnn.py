import keras

class RNN():

    def __init__(self, feat_size, timesteps, layers=1, embedding_layer=False, embedding_size=100):

        #input layer
        input = keras.layers.Input(shape=(timesteps, feat_size))

        '''if embedding_layer is True:
            input = keras.layers.Embedding(input_dim=(timesteps, feat_size), output_dim=(timesteps, embedding_size))(input)'''


        # Batch normalization for input normalization
        rnn = keras.layers.BatchNormalization(axis=-1)(input)
        #rnn = input

        #rnn layers
        for i in range(layers):

            return_sequences = True
            if i==layers-1:
                return_sequences = False

            rnn = keras.layers.SimpleRNN(units=32, activation='relu', return_sequences=return_sequences, kernel_regularizer=keras.regularizers.l2(0.5))(rnn)


        #Batch normalization
        #out = keras.layers.BatchNormalization(axis=-1)(rnn)

        out = keras.layers.Dropout(rate=0.5)(rnn)

        #output
        out = keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.5))(out)
        #out = keras.layers.Dense(2)(out)
        out = keras.layers.Activation('softmax')(out)

        #Model
        self.model = keras.models.Model(inputs=[input], outputs=[out])


    def get_model(self):
        return self.model


class BasicRNN():
    def __init__(self, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        self.model.add(keras.layers.BatchNormalization())
        # RNN
        self.model.add(keras.layers.SimpleRNN(units=embedding_size, activation='relu', kernel_regularizer=keras.regularizers.l2(1)))
        self.model.add(keras.layers.Dropout(rate=0.5))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(1)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class GRURNN():
    def __init__(self, embedding_layer=True, embedding_size=100, **kwargs):
        self.model = keras.Sequential()

        self.model.add(keras.layers.BatchNormalization())

        #input layer
        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(0.1)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.GRU(units=embedding_size, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(keras.layers.Dropout(rate=0.5))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class LSTMRNN():
    def __init__(self, embedding_layer=True, embedding_size=100):
        self.model = keras.Sequential()

        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(0.5)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.LSTM(units=embedding_size, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.5)))
        self.model.add(keras.layers.Dropout(rate=0.5))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.5)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model

class GRU2RNN():
    def __init__(self, timesteps, embedding_layer=True, embedding_size=100):
        self.model = keras.Sequential()

        if embedding_layer:
            self.model.add(keras.layers.Dense(units=embedding_size, kernel_regularizer=keras.regularizers.l2(0.5)))
            self.model.add(keras.layers.Activation('relu'))

        # RNN
        self.model.add(keras.layers.GRU(units=embedding_size, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.5), return_sequence=True))
        self.model.add(keras.layers.Dropout(rate=0.5))
        self.model.add(keras.layers.GRU(units=embedding_size, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.5), return_sequence=False))
        self.model.add(keras.layers.Dropout(rate=0.5))

        #output
        self.model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.5)))
        self.model.add(keras.layers.Activation('softmax'))


    def get_model(self):
        return self.model
