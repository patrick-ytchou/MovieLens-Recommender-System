import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

def load_data():
    """
    Load in dataset.
    """
    A = load_npz("data/Atrain.npz")
    A_test = load_npz("data/Atest.npz")
    mask = (A > 0) * 1.0
    mask_test = (A_test > 0) * 1.0
    
    # make copies since we will shuffle
    A_copy = A.copy()
    mask_copy = mask.copy()
    A_test_copy = A_test.copy()
    mask_test_copy = mask_test.copy()
    
    N, M = A.shape
    mu = A.sum() / mask.sum()
    return A, mask, A_test, mask_test, A_copy, mask_copy, A_test_copy, mask_test_copy

def initialize_variables(batch_size, epochs, reg):
    return batch_size, epochs, reg


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


def generator(A, M, batch_size, mu):
    """
    Generator function for the train data for model.fit_generator 
    """
    while True:
        # mu = A.sum() / M.sum()
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
            
def test_generator(A, M, A_test, M_test, batch_size, mu):
    """
    Generator function for the test data for model.fit_generator
    """
    while True:
        loop_size = A.shape[0] // batch_size + 1
        for i in range(loop_size):
            upper = min((i+1)*batch_size, A.shape[0])
            a = A[i*batch_size:upper].toarray()
            m = M[i*batch_size:upper].toarray()
            at = A_test[i*batch_size:upper].toarray()
            mt = M_test[i*batch_size:upper].toarray()
            a = a - mu * m
            at = at - mu * mt
            yield a, at


def train_model(dropout_perc, dense_layer_size, batch_size, epochs, reg, A, mask, A_copy, mask_copy, A_test_copy, mask_test_copy):
    """
    Build neural network model.
    """
    # Construct model
    N, M = A_copy.shape 
    mu = A_copy.sum() / mask_copy.sum()
    
    i = Input(shape=(M,))
    x = Dropout(dropout_perc)(i)
    x = Dense(dense_layer_size, activation='relu', kernel_regularizer=l2(reg))(x)
    # x = Dropout(dropout_perc*0.8)(x)
    x = Dense(M, kernel_regularizer=l2(reg))(x)

    # Compile and fit the model
    model = Model(i, x)
    model.compile(
        loss = custom_loss,
        optimizer=SGD(lr=0.08, momentum=0.9),
        # optimizer = Adam(learning_rate = 0.08),
        metrics = [custom_loss]
    )
    
    r = model.fit(
        generator(A, mask, batch_size, mu),
        validation_data = test_generator(A_copy, mask_copy, A_test_copy, mask_test_copy, batch_size, mu),
        epochs=epochs,
        steps_per_epoch=A.shape[0] // batch_size + 1,
        validation_steps=A_test.shape[0] // batch_size + 1,
    )
    
    print(r.history.keys())
    
    return model, r

def plot_result(r):
    """
    Plot the result.
    """
    # plot losses
    plt.plot(r.history['loss'], label = 'Train Loss')
    plt.plot(r.history['val_loss'], label = 'Test Loss')
    plt.legend()
    plt.show()
    
    # plot mse
    plt.plot(r.history['custom_loss'], label = 'Train MSE')
    plt.plot(r.history['val_custom_loss'], label = 'Test MSE')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print("Now: Initialize variables.")
    batch_size, epochs, reg = initialize_variables(batch_size = 256, epochs = 2, reg = 0.001)
    print("Now: Load Dataset.")
    A, mask, A_test, mask_test, A_copy, mask_copy, A_test_copy, mask_test_copy = load_data()
    print("Now: Train model")
    model, r = train_model(dropout_perc = 0.7, 
                dense_layer_size = 600, 
                batch_size = batch_size, 
                epochs = epochs, 
                reg = reg, 
                A = A, 
                mask = mask,
                A_copy= A_copy,
                mask_copy = mask_copy,
                A_test_copy = A_test_copy, 
                mask_test_copy = mask_test_copy)
    print("Now: Save Model.")
    model.save("model/autorec_batch_256.h5")
    plot_result(r)
    