import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model, load_model
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


def predict(model, user_input):
    """
    Build neural network model.
    """
    user = pd.Series([18610] * 26744)
    movie = pd.Series(list(range(26744)))
    
    result = model.predict(
        x = [user, movie],
        # x = [user_input.userId.values, user_input.movie_idx.values],
        verbose = 1
        )
    result = np.array(result).flatten()
    print(result)
    return result

if __name__ == "__main__":

    print("Now: Load Model")
    model = load_model("model/mf_deep.h5")

    print("Now: Load user input.")
    user_input = load_data("data/sample_input.csv")
    print(user_input.shape)
    
    print("Now: Generate movie recommendation based on user input.")    
    prediction = predict(model, user_input)
    
    print("Top 15 indexes")
    top_2_idx = np.argsort(prediction)[-15:]
    print(top_2_idx)
    top_2_values = [prediction[i] for i in top_2_idx]
    print(top_2_values)