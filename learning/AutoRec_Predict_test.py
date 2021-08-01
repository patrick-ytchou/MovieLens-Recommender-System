import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

def load_data(input_filepath):
    """
    Load in dataset.
    """
    npz = load_npz(input_filepath)
    return npz


def generator(A, M):
    """
    Generator function for the train data for model.fit_generator 
    """
    while True:
        batch_size = 256
        mu = A.sum() / M.sum()
        A, M = shuffle(A, M)
        loop_size = A.shape[0] // batch_size + 1
        for i in range(loop_size):
            upper = min((i+1)*batch_size, A.shape[0])
            a = A[i*batch_size:upper].toarray()
            m = M[i*batch_size:upper].toarray()
            a = a - mu * m  # center the data
            m2 = (np.random.random(a.shape) > 0.5) # noise term
            noisy = a * m2 
            print(i)
            print(noisy.shape, a.shape)
            yield noisy, a            

def predict(model, user_input):
    """
    Build neural network model.
    """
    mask  = (user_input > 0) * 1.0
    result = model.predict(
        generator(user_input, mask),
        verbose = 1
        )
    print(result)
    print(result.shape)
    return result


def custom_loss(y_true, y_pred):
    """
    Build custom MSE loss function for autoencoder.
    """
    ## find y_true that are not zero --> there are data for prediction
    mask = K.cast(K.not_equal(y_true, 0), dtype='float')
    diff = y_pred - y_true
    sqdiff = diff * diff * mask ## those being masked are not considered
    sse = K.sum(K.sum(sqdiff)) ## sum all the difference
    n = K.sum(K.sum(mask)) ## sum of the entries
    return sse / n


if __name__ == '__main__':
    print("Now: Load model.")
    y_true = np.array([1,1,1])
    y_pred = np.array([1,1,1])
    model = load_model("model/autorec_batch_256.h5", custom_objects = {"custom_loss": custom_loss(y_true, y_pred)})
    
    print("Now: Load user input.")
    user_input = load_data("data/sample_input.npz")
    print(user_input.shape)
    
    # test = load_data("data/Atrain.npz")
    # print(test.shape)
    
    print("Now: Generate movie recommendation based on user input.")    
    prediction = predict(model, user_input)
    