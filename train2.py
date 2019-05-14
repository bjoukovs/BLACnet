import keras
from model.rnn import RNN
import numpy as np
import utils.utils as u

#### DATA ####

train = np.load('feature_extraction/output/featuresTensor_train_1000.npy')
val = np.load('feature_extraction/output/featuresTensor_val_1000.npy')

K = 1000
N = 12

# Extracting inputs (ndarray of shape (samples, timesteps, nfeatures)
inputs_train = u.format_inputs(train[:, 0], N, K)
inputs_val = u.format_inputs(val[:, 0], N, K)

# Extracting labels (ndarray of size (samples, 1))
labels_train = np.array(train[:, 1])
labels_val = np.array(val[:, 1])

print(len(np.where(labels_train==1)[0]))
print(len(np.where(labels_train==0)[0]))
print(len(np.where(labels_val==1)[0]))
print(len(np.where(labels_val==0)[0]))


# NOTE : THE INPUT DATA SHOULD BE NORMALIZED SOMEHOW



#### MODEL ####
BATCH_SIZE = 16
NAME = "RNN"

model = RNN(feat_size=K, timesteps=N, layers=1, embedding_layer=False).get_model()
print(model.summary())


#### Optimizer ####
optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy, keras.metrics.binary_crossentropy])

#### Callbacks ####
checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/'+NAME+'.hdf5', monitor='val_loss', save_best_only=True)


#### Train ####
model.fit(x = inputs_train, y = labels_train, validation_data = (inputs_val, labels_val), batch_size=BATCH_SIZE, epochs=100, verbose=2, callbacks=[checkpoint])



