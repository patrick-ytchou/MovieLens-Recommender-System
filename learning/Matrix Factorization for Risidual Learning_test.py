import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

def load_data(filepath):
    """
    Load data for prediction.
    """
    df = pd.read_csv(filepath)
    return df

def preprocessing(df, train_size):
    """
    Conduct preprocessing to the data for later modeling.
    """
    N = df.userId.max() + 1 # num of users
    M = df.movie_idx.max() + 1 # num of movies
    
    # split data into train and test
    df = shuffle(df)
    cutoff_point = int(train_size*len(df))
    df_train = df.iloc[:cutoff_point]
    df_test = df.iloc[cutoff_point:]
    return N, M, df_train, df_test

def modeling(df_train, df_test, N, M, epochs, batch_size, dim, reg):
    """
    Conduct modeling in Keras.
    """
    # compute global mean
    mu = df_train.rating.mean()
    
    # modeling
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    u_embedding = Embedding(N, dim, embeddings_regularizer=l2(reg))(u) # (N, 1, dim)
    m_embedding = Embedding(M, dim, embeddings_regularizer=l2(reg))(m) # (M, 1, dim)
    u_embedding = Flatten()(u_embedding) # (N, dim)
    m_embedding = Flatten()(m_embedding) # (N, dim)
    x = Concatenate()([u_embedding, m_embedding]) # (N, 2*dim)
    x = Dense(400)(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(100)(x)
    # x = BatchNormalization()(x)ç
    x = Activation('relu')(x)
    x = Dense(1)(x)
    
    model = Model(inputs = [u, m], outputs = x)
    model.compile(
        loss = 'mse',
        optimizer = Adam(learning_rate=0.01),
        metrics = ['mse']
    )
    
    r = model.fit(  
        x = [df_train.userId.values, df_train.movie_idx.values],
        y = df_train.rating.values - mu,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = [
            [df_test.userId.values, df_test.movie_idx.values],
            df_test.rating.values - mu
        ]
    )
    return r

def plot_result(model):
    """
    Plot the mode result.
    """
    # plot losses
    plt.plot(model.history['loss'], label='train loss')
    plt.plot(model.history['val_loss'], label = 'test loss')
    plt.legend()
    plt.show()
    
    # plot mse
    plt.plot(model.history['mean_squared_error'], label='train mse')
    plt.plot(model.history['val_squared_error'], label = 'test mse')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Now: Load data.")
    df = load_data("data/edited_rating.csv")
    print("Now: Implement preprocessing.")
    N, M, df_train, df_test = preprocessing(df, train_size = .8)
    print("Now: Model training.")
    model = modeling(df_train, df_test, N, M, epochs = 3, batch_size = 128, dim = 10, reg = 0.1)
    plot_result(model)