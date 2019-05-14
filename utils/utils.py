import numpy as np

def pad_events_with_zeroes(event, N, K):

    size = event.shape
    event_timesteps = size[0]

    output = np.zeros(shape=(N, K))

    #check if not enough or enough timesteps: then pad with zeroes
    if event_timesteps <= N:
        output[N-event_timesteps:N, :] = event

    #otherwise trim
    else:
        output[:, :] = event[:N, :]


    return output



def format_inputs(inputs, N, K):

    dataset_length = len(inputs)

    input_tensor = np.empty(shape=(dataset_length, N, K))

    idx = 0
    for input in inputs:

        #Transpose input
        input = np.rollaxis(input, 1, 0)
        input_tensor[idx, ] = pad_events_with_zeroes(input, N, K)
        idx += 1

    return input_tensor

