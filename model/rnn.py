import keras

class RNN():

    def __init__(self, feat_size, timesteps, layers=1, embedding_layer=False, embedding_size=100):

        #input layer
        input = keras.layers.Input(shape=(timesteps, feat_size))

        '''if embedding_layer is True:
            input = keras.layers.Embedding(input_dim=(timesteps, feat_size), output_dim=(timesteps, embedding_size))(input)'''


        rnn = input

        #rnn layers
        for i in range(layers):

            return_sequences = True
            if i==layers:
                return_sequences = False

            rnn = keras.layers.SimpleRNN(units=feat_size, activation='tanh', return_sequences=return_sequences)(rnn)


        #output
        out = keras.layers.Dense(2)(rnn)
        out = keras.layers.Activation('softmax')(rnn)

        #Model
        self.model = keras.models.Model(inputs=[input], outputs=[out])


    def get_model(self):
        return self.model







