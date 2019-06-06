import numpy as np

def pad_events_with_zeroes(event, N, K, pca=None):

    size = event.shape
    event_timesteps = size[0]

    output = np.zeros(shape=(N, K))

    #check if not enough or enough timesteps: then pad with zeroes

    if pca is not None:
        feature_vector = pca.transform(event)
    else:
        feature_vector = event

    # check if not enough or enough timesteps: then pad with zeroes
    if event_timesteps <= N:
        output[:event_timesteps, :] = feature_vector

    # otherwise trim
    else:
        output[:, :] = feature_vector[:N, :]


    return output



def format_inputs(inputs, N, K, pca=None):

    dataset_length = len(inputs)

    input_tensor = np.empty(shape=(dataset_length, N, K))

    idx = 0
    for input in inputs:

        #Transpose input
        input = np.rollaxis(input, 1, 0)
        input_tensor[idx, ] = pad_events_with_zeroes(input, N, K, pca)
        idx += 1

    return input_tensor

def format_inputs_notime(inputs):

    dataset_length = len(inputs)
    vectors = []
    labels = []
    for input in inputs:

        vectors.append(input[0].T)
        labels.append(input[1])

    return np.array(vectors), np.array(labels)



def majority_voting(predictions, labels):
    #predictions = predictions > 0.5
    predictions = np.sum(predictions, axis=0)
    predictions = np.argmax(predictions, axis=-1)

    correct = np.bincount(predictions == np.argmax(labels, axis=-1))[1]
    accuracy = correct/labels.shape[0]

    return(accuracy)

