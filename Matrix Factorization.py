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
    
def get_loss(d):
    """
    Calculate loss for each user movie rating.
    
    Parameters
    --------
    d: (user_id, movie_id) -> rating
    """
    N = float(len(d))
    sse = 0 
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p-r)**2
    return sse / N
    
def train(epochs, reg, N, M ,K, W, b, U, c, mu, usermovie2rating, usermovie2rating_test):
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
            # for W
            matrix = np.eye(K) * reg
            vector = np.zeros(K)
            
            # for b
            bi = 0
            for j in user2movie[i]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(U[j], U[j])
                vector += (r - b[i] - c[j] - mu) * U[j]
                bi += (r - W[i].dot(U[j]) - c[j] - mu)
            
            # set the updates
            W[i] = np.linalg.solve(matrix, vector)
            b[i] = bi / (len(user2movie[i]) + reg)
            
            if i % (N//10) == 0:
                print("Now i: ", i, ", N: ", N)
        print("Now: Update W and b: ", datetime.now() - t0)

        
        # update U and c
        t0 = datetime.now()
        for j in range(M):
            # for U
            matrix = np.eye(K) * reg
            vector = np.zeros(K)
            
            # for c
            cj = 0
            try:
                for i in movie2user[j]:
                    r = usermovie2rating[(i, j)]
                    matrix += np.outer(W[i], W[i])
                    vector += (r - b[i] - c[j] - mu) * W[i]
                    cj += (r - W[i].dot(U[j]) - b[i] - mu)
                    
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
        train_losses.append(get_loss(usermovie2rating))
        
        # store test loss
        test_losses.append(get_loss(usermovie2rating_test))
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
        
    # return train_losses, test_losses, W, b, U, c



if __name__ == '__main__':
    print("Now: Load data.")
    user2movie, movie2user, usermovie2rating, user2movie_test, movie2user_test, usermovie2rating_test = load_data()
    N, M, K, W, b, U, c, mu = initialize_variables(dim = 5, 
                                             user2movie = user2movie, 
                                             movie2user = movie2user, 
                                             movie2user_test = movie2user_test, 
                                             usermovie2rating = usermovie2rating)
    train(epochs = 15, reg = 20, N = N, M = M, K = K, W = W, 
          b = b, U = U, c = c, mu = mu, 
          usermovie2rating = usermovie2rating, 
          usermovie2rating_test= usermovie2rating_test)