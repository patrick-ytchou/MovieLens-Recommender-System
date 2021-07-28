import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

def load_data():
    """
    Load in the pickle file.
    """
    try:
        with open('data/user2movie.json', 'rb') as f:
            user2movie = pickle.load(f)
        with open('data/movie2user.json', 'rb') as f:
            movie2user = pickle.load(f)
        with open('data/usermovie2rating.json', 'rb') as f:
            usermovie2rating = pickle.load(f)
        with open('data/user2movie_test.json', 'rb') as f:
            user2movie_test = pickle.load(f)
        with open('data/movie2user_test.json', 'rb') as f:
            movie2user_test = pickle.load(f)
        with open('data/usermovie2rating_test.json', 'rb') as f:
            usermovie2rating_test = pickle.load(f)
    except:
        raise Exception('File does not exist.')
    
    return user2movie, movie2user, usermovie2rating, user2movie_test, movie2user_test, usermovie2rating_test

def convert_dataset(user2movie, movie2user, usermovie2rating_test):
    """
    Convert the dataset to make it vectorized.
    """
    user2movierating = {}
    for i, movies in user2movie.items():
        r = np.array([usermovie2rating[(i,j)] for j in movies])
        user2movierating[i] = (movies, r)
    movie2userrating = {}
    for j, users in movie2user.items():
        r = np.array([usermovie2rating[(i,j)] for i in users])
        movie2userrating[j] = (users, r)
    
    movie2userrating_test = {}
    for (i, j), r in usermovie2rating_test.items():
        if j not in movie2userrating_test:
            movie2userrating_test[j] = [[i], [r]]
        else:
            movie2userrating_test[j][0].append(i)
            movie2userrating_test[j][1].append(r)
    for j, (users, r) in movie2userrating_test.items():
        movie2userrating_test[j][1] = np.array(r)
    return user2movierating, movie2userrating, movie2userrating_test



def initialize_variables(dim, user2movie, movie2user, movie2user_test, usermovie2rating):
    """
    Initialize variables for training.
    """
    N = np.max(list(user2movie.keys())) + 1
    m1 = np.max(list(movie2user.keys()))
    m2 = np.max(list(movie2user_test.keys()))
    M = max(m1, m2) + 1
    
    K = dim
    W = np.random.rand(N,K)
    b = np.zeros(N)
    U = np.random.rand(M,K)
    c = np.zeros(M)
    mu = np.mean(list(usermovie2rating.values()))
    return N, M, K, W, b, U, c, mu
    
def get_loss(m2u):
    """
    Calculate loss for each user movie rating.
    
    Parameters
    --------
    m2u: movie_id -> (user_ids, ratings)
    """
    N = 0.
    sse = 0
    for j, (u_ids, r) in m2u.items():
        p = W[u_ids].dot(U[j]) + b[u_ids] + c[j] + mu
        delta = p - r
        sse += delta.dot(delta)
        N += len(r)
    return sse / N


def train(epochs, reg, N, M ,K, W, b, U, c, mu, movie2userrating, movie2userrating_test):
    """
    Train parameters.
    
    Parameters
    --------
    epochs: int, times to run
    reg: float, regularization penalty
    """
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print("Now: Epoch ", epoch)
        epoch_start = datetime.now()
        
        # update W and b
        t0 = datetime.now()
        for i in range(N):
            m_ids, r = user2movierating[i]
            matrix = U[m_ids].T.dot(U[m_ids]) * np.eye(K) * reg
            vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])
            bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()
            
            # set the updates
            W[i] = np.linalg.solve(matrix, vector)
            b[i] = bi / (len(user2movie[i]) + reg)
            
            if i % (N//10) == 0:
                print("Now i: ", i, ", N: ", N)
        print("Now: Update W and b: ", datetime.now() - t0)

        
        # update U and c
        t0 = datetime.now()
        for j in range(M):
            try:
                u_ids, r = movie2userrating[j]
                matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K) * reg
                vector = (r - b[u_ids] - c[j] - mu).dot(W[u_ids])
                cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - mu).sum()
                    
                # set the updates
                U[j] = np.linalg.solve(matrix, vector)
                c[j] = cj / (len(movie2user[j]) + reg)
                
                if j % (M//10) == 0:
                    print("Now j: ", j, ", M: ", M)
            except KeyError:
                # pass for movies that don't have any rating
                pass
        print("Now: Update U and c: ", datetime.now() - t0)
        print("Epoch duration: ", datetime.now() - epoch_start)
        
        # store train loss
        t0 = datetime.now()
        train_losses.append(get_loss(movie2userrating))
        
        # store test loss
        test_losses.append(get_loss(movie2userrating_test))
        print("Calculate cost: ", datetime.now() - t0)
        print("Train loss: ", train_losses[-1])
        print("Test loss: ", test_losses[-1])
    
    print("Train losses: ", train_losses)
    print("Train losses: ", test_losses)     
    
    # plot losses
    plt.plot(train_losses, label = "train loss")
    plt.plot(test_losses, label = "test loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print("Now: Load data.")
    user2movie, movie2user, usermovie2rating, user2movie_test, movie2user_test, usermovie2rating_test = load_data()
    user2movierating, movie2userrating, movie2userrating_test = convert_dataset(user2movie, movie2user, usermovie2rating_test)
    N, M, K, W, b, U, c, mu = initialize_variables(dim = 5, 
                                             user2movie = user2movie, 
                                             movie2user = movie2user, 
                                             movie2user_test = movie2user_test, 
                                             usermovie2rating = usermovie2rating)
    train(epochs = 15, reg = 20, N = N, M = M, K = K, W = W, 
          b = b, U = U, c = c, mu = mu, 
          movie2userrating  = movie2userrating, 
          movie2userrating_test = movie2userrating_test)