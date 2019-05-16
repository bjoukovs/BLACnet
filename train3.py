import keras
import numpy as np
import utils.utils as u
import sklearn.decomposition as skd

#### DATA ####

train = np.load('feature_extraction/output2/featuresTensor_train.npy')
val = np.load('feature_extraction/output2/featuresTensor_val.npy')

K = 2500

#Getting all feature vectors for PCA
#all_inputs_train = u.format_inputs_notime(train[:, 0])
#pca = skd.SparsePCA(n_components=K)
#pca.fit(all_inputs_train)
pca = None

inputs_train, labels_train = u.format_inputs_notime(train)
inputs_train = np.squeeze(inputs_train, axis=-1)

inputs_val, labels_val = u.format_inputs_notime(val)
inputs_val = np.squeeze(inputs_val, axis=-1)



print(len(np.where(labels_train==1)[0]))
print(len(np.where(labels_train==0)[0]))
print(len(np.where(labels_val==1)[0]))
print(len(np.where(labels_val==0)[0]))

labels_train = keras.utils.to_categorical(labels_train, num_classes=2)
labels_val = keras.utils.to_categorical(labels_val, num_classes=2)

# NOTE : THE INPUT DATA SHOULD BE NORMALIZED SOMEHOW

a = train[0,0]

#### MODEL ####
BATCH_SIZE = 32
NAME = "NN14"


model = keras.Sequential()
model.add(keras.layers.BatchNormalization(batch_input_shape=(None,K)))
model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(2, kernel_regularizer=keras.regularizers.l2(0.1)))
model.add(keras.layers.Activation('softmax'))



#### Optimizer ####
optimizer = keras.optimizers.RMSprop(lr=1e-4)

model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.categorical_accuracy, keras.metrics.categorical_crossentropy])

#### Callbacks ####
#checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/'+NAME+'.hdf5', monitor='val_loss', save_best_only=True)
tb = keras.callbacks.TensorBoard('logs/'+NAME)
#lrshedule = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_lr=5e-7, verbose=1)


#### Train ####
model.fit(x = inputs_train, y = labels_train, validation_data = (inputs_val, labels_val), batch_size=BATCH_SIZE, epochs=250, verbose=2, class_weight=None, callbacks=[tb], shuffle=True)


print(model.evaluate(x = inputs_val, y = labels_val, batch_size=BATCH_SIZE, verbose=2))


