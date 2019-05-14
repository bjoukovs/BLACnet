import keras

class RNN():

    def __init__(self, feat_size, timesteps, layers=1, embedding_layer=False, embedding_size=100):

        #input layer
        input = keras.layers.Input(shape=(timesteps, feat_size))

        '''if embedding_layer is True:
            input = keras.layers.Embedding(input_dim=(timesteps, feat_size), output_dim=(timesteps, embedding_size))(input)'''


        # Batch normalization for input normalization
        #rnn = keras.layers.BatchNormalization(axis=-1)(input)
        rnn = input

        #rnn layers
        for i in range(layers):

            return_sequences = True
            if i==layers-1:
                return_sequences = False

            rnn = keras.layers.SimpleRNN(units=feat_size, activation='relu', return_sequences=return_sequences)(rnn)


        #Batch normalization
        out = keras.layers.BatchNormalization(axis=-1)(rnn)

        #output
        out = keras.layers.Dense(2)(out)
        out = keras.layers.Activation('softmax')(out)

        #Model
        self.model = keras.models.Model(inputs=[input], outputs=[out])


    def get_model(self):
        return self.model







