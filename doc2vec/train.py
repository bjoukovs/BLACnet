import keras
import numpy as np


train_x = np.load('train_x.npy')
val_x = np.load('val_x.npy')

train_y = keras.utils.to_categorical(np.load('train_y.npy'))
val_y = keras.utils.to_categorical(np.load('val_y.npy'))


model = keras.Sequential()
model.add(keras.layers.BatchNormalization(batch_input_shape=(None,50)))
model.add(keras.layers.Dense(50))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation('softmax'))

model.summary()

model.compile(optimizer=keras.optimizers.Adagrad(), loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.categorical_crossentropy, keras.metrics.categorical_accuracy])

model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), batch_size=64, epochs=300, verbose=2)