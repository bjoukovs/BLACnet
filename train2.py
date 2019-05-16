import keras
from model.rnn import RNN
import numpy as np
import utils.utils as u
import sklearn.decomposition as skd

#### DATA ####

train = np.load('feature_extraction/output/featuresTensor_train.npy')
val = np.load('feature_extraction/output/featuresTensor_val.npy')

K = 1000
N = 12

#Getting all feature vectors for PCA
#all_inputs_train = u.format_inputs_notime(train[:, 0])
#pca = skd.SparsePCA(n_components=K)
#pca.fit(all_inputs_train)
pca = None

# Extracting inputs (ndarray of shape (samples, timesteps, nfeatures)
inputs_train = u.format_inputs(train[:, 0], N, K, pca)
inputs_val = u.format_inputs(val[:, 0], N, K, pca)



# Extracting labels (ndarray of size (samples, 1))
labels_train = np.array(train[:, 1])
labels_val = np.array(val[:, 1])

print(len(np.where(labels_train==1)[0]))
print(len(np.where(labels_train==0)[0]))
print(len(np.where(labels_val==1)[0]))
print(len(np.where(labels_val==0)[0]))

class_weight = {0:2.2/4,
                1:1.8/4}

labels_train = keras.utils.to_categorical(labels_train, num_classes=2)
labels_val = keras.utils.to_categorical(labels_val, num_classes=2)

# NOTE : THE INPUT DATA SHOULD BE NORMALIZED SOMEHOW

a = train[0,0]

#### MODEL ####
BATCH_SIZE = 32
NAME = "RNN22"

model = RNN(feat_size=K, timesteps=N, layers=1, embedding_layer=False).get_model()
print(model.summary())



#### Optimizer ####
optimizer = keras.optimizers.RMSprop(lr=1e-4)

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy])

#### Callbacks ####
#checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/'+NAME+'.hdf5', monitor='val_loss', save_best_only=True)
tb = keras.callbacks.TensorBoard('logs/'+NAME)
#lrshedule = keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.5, patience=5, min_lr=1e-5)


#### Train ####
model.fit(x = inputs_train, y = labels_train, validation_data = (inputs_val, labels_val), batch_size=BATCH_SIZE, epochs=150, verbose=2, class_weight=None, callbacks=[tb], shuffle=True)


model.evaluate(x = inputs_val, y = labels_val, batch_size=BATCH_SIZE)


