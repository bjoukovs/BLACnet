import keras
from model.rnn import RNN

BATCH_SIZE = 10
NAME = "RNN"

model = RNN(100, 10, layers=1, embedding_layer=False).get_model()
print(model.summary())

#### DATA ####


#### Optimizer ####
optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.categorical_accuracy, keras.metrics.binary_crossentropy])

#### Callbacks ####
checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/'+NAME+'.hdf5', monitor='val_loss', save_best_only=True)


#### Train ####
model.fit(x=[], y=[], batch_size=BATCH_SIZE, epochs=100, verbose=1, callbacks=[checkpoint])



