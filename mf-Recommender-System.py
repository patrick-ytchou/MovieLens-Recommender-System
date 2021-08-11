import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dot, Add, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

def load_data(filepath):
    """
    Load data for prediction.
    """
    df = pd.read_csv(filepath)
    return df

def preprocessing(df):
    """
    Proprocess the data accordingly.
    """
    
    # # make userid goes from 0...N-1
    # df.user_id = df.user_id - 1
    
    # create a new mapping for business id to create a continuous ids    
    unique_business_ids = set(df.business_id.values)
    buisness2idx = {}
    new_id = 0
    for old_id in unique_business_ids:
        buisness2idx[old_id] = new_id
        new_id += 1    
        
    
    # create a new mapping for user id to create a continuous ids
    unique_user_ids = set(df.user_id.values)
    user2idx = {}
    new_id = 0
    for old_id in unique_user_ids:
        user2idx[old_id] = new_id
        new_id += 1
    
    df['business_idx'] = df.apply(lambda row: buisness2idx[row.business_id], axis=1)
    df['user_idx'] = df.apply(lambda row: user2idx[row.user_id], axis=1)
    # df = df.drop(columns=['timestamp'])

    # df = df[['user_idx', 'business_idx', 'stars']]
    df['stars'] = df['stars'].apply(lambda x: float(x))    

    df.to_csv('data/edited_review.csv')
    return df



def prep_for_modeling(df, train_size):
    """
    Conduct preprocessing to the data for later modeling.
    """
    N = len(list(set(df.user_idx))) + 1 # num of users
    M = len(list(set(df.business_idx))) + 1 # num of movies
    print(N, M)
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
    # mu = df_train.rating.mean()
    
    # modeling
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    
    # main branch -- matrix factorization
    u_embedding = Embedding(N, dim, embeddings_regularizer=l2(reg))(u) # (N, 1, dim)
    m_embedding = Embedding(M, dim, embeddings_regularizer=l2(reg))(m) # (M, 1, dim)
    
    u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
    m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (M, 1, 1)
    x = Dot(axes=2)([u_embedding, m_embedding])
    x = Add()([x, u_bias, m_bias])
    x = Flatten()(x) # (N, 1)

    # side branch -- deep learning
    u_embedding = Flatten()(u_embedding) # (N, dim)
    m_embedding = Flatten()(m_embedding) # (N, dim)
    y = Concatenate()([u_embedding, m_embedding]) # (N, 2*dim)
    y = Dense(400)(y)
    y = Activation('relu')(y)
    y = Dropout(0.3)(y)
    y = Dense(100)(y)
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = Dense(1)(x)
    
    # concat two branches together
    x = Add()([x, y])
    
    
    model = Model(inputs = [u, m], outputs = x)
    
    model.compile(
        loss = 'mse',
        optimizer = Adam(learning_rate=0.01),
        metrics = ['mse']
    )
    
    print(model.summary())
    
    r = model.fit(  
        x = [df_train.user_idx.values, df_train.business_idx.values],
        y = df_train.stars.values,        
        # y = df_train.rating.values - mu,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (
            [df_test.user_idx.values, df_test.business_idx.values],
            # df_test.rating.values - mu
            df_test.stars.values
        )
    )
    return model, r

if __name__ == "__main__":
    print("Now: Load data.")
    df = load_data("data/review.csv")
    print("Now: Implement preprocessing.")
    df = preprocessing(df)
    # df = df[['user_idx', 'business_idx', 'stars']]
    N, M, df_train, df_test = prep_for_modeling(df, train_size = .8)
    print("Now: Model training.")
    model, r = modeling(df_train, df_test, N, M, epochs = 5, batch_size = 2048, dim = 30, reg = 0.1)
    # print("Now: Save Model.")
    # model.save("model/model_v1.h5")
    # print("Now: Save Weights.")
    # model.save_weights("model/model_weight_v1.h5")