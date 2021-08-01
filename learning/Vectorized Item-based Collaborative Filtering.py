import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from itertools import permutations

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





def _compute(i, usermovie2rating, user_set):
    """
    Helper function to compute coefficient.
    """
    
    rating = {user: usermovie2rating[(user, i)] for user in user_set}
    avg = np.mean(list(rating.values()))
    dev = {user: (rating - avg) for user, rating in rating.items()}
    
    dev_values = np.array(list(dev.values()))
    sigma = np.sqrt(dev_values.dot(dev_values))

    return rating, avg, sigma, dev

def calculate_coef(K, limit, user2movie, movie2user, movie2user_test):
    """
    Conduct modeling.
    
    Parameters
    --------
    K: int, number of neighbors to consider for correlation calculation
    limit: int, least number of common movies users mush have in common in order to consider
    """
    N = np.max(list(user2movie.keys())) + 1
    m1 = np.max(list(movie2user.keys()))
    m2 = np.max(list(movie2user_test.keys())) 
    M = max(m1, m2) + 1 ## we might find another unseen movie id in test set.
    
    neighbors = [] # store neighbors in this list 
    averages_i = [] # each user's average rating
    deviations_i = [] # each user's deviation
    ratings_i = []
    sigmas_i = []
    ui_sets = []
    i = 0
    print("Total movies: ", M)
    for i in range(M):
        print("Now processing: movie ", i)
        users_i = movie2user[i]
        users_i_set = set(users_i)
        r, a, s, d = _compute(i, usermovie2rating, users_i_set)

        # save results
        ratings_i.append(r)
        averages_i.append(a)
        deviations_i.append(d)
        sigmas_i.append(s)
        ui_sets.append(users_i_set)
        
    ## Create common movie set
    common_movies = []
    linspace = np.linspace(0, M-1, M, dtype=int)
    for i, j in list(permutations(linspace, 2)):
        if i != j:
            cm = (ui_sets[i], ui_sets[j]) # intersection
            common_movies.append(cm)
    
    # averages_j = [] # each user's average rating
    # deviations_j = [] # each user's deviation
    # ratings_j = []
    # sigmas_j = []
    # for j in range(M):
    #     # if len(common_movies) > limit:
    #     # calculate avg and deviation
    #     r, a, s, d, cm = _compute(i, usermovie2rating)
        
    #     # save results
    #     ratings_j.append(r)
    #     averages_j.append(a)
    #     deviations_j.append(d)
    #     sigmas_j.append(s)
        # common_movies.append(cm)
    
    # linspace = np.linspace(0, M-1, M, dtype=int)
    sl = []
    for x, y in list(permutations(linspace, 2)):
        # calculate correlation coefficient
        if x != y and len(common_movies[y]) > limit:
            print(common_movies[y])
            print(deviations_i[x])
            print(deviations_i[y])
            numerator = sum(deviations_i[x][m]*deviations_i[y][m] for m in common_movies[y])
            w_ij = numerator / (sigmas_i[x] * sigmas_i[y])
            
            # truncate if there are too many values in the sorted list
            # sl.add((-w_ij,j)) # store negative weight because the list is sorted ascending
            sl.append((-w_ij, j))
            sl = sorted(sl)
            if len(sl) > K:
                del sl[-1]

        neighbors.append(sl)
    return sl, neighbors, averages_i, deviations_i
 
def _predict(i, m, sl, neighbors, averages, deviations):
    """
    Helper function to make prediction for user i on movie m based on pre-computed coef.
    
    Parameters
    --------
    i: int, index for user 
    m: int, index for movie
    sl: sorted list
    neighbors: 
    averages: 
    deviations: 
    """
    numerator, denominator = 0, 0
    for neg_w, j in neighbors[i]:
        try:
            # note that we store negative weights
            numerator += -neg_w * deviations[j][m]
            denominator += abs(-neg_w)
        except KeyError: # if the movie does not exist
            pass
    
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
        
    # clip the prediction to [0.5, 5]
    prediction = max(min(5, prediction), 0.5)
    return prediction

def predict(usermovie2rating, usermovie2rating_test, sl, neighbors, averages, deviations):
    """
    Make prediction for all the ratings.
    """
    train_predictions = []
    train_targets = []
    print("Now: Loop through training dataset.")
    i = 0 
    for (i, m), target in usermovie2rating.items():
        print("Now predicting (train): user ", i)
        # predict for each of the user movie rating entry
        prediction = _predict(m, i, sl, neighbors, averages, deviations)
        
        train_predictions.append(prediction)
        train_targets.append(target)
        i += 1
    
    print("Now: Loop through testing dataset.")
    test_predictions = []
    test_targets = []
    i = 0
    for (i, m), target in usermovie2rating_test.items():
        print("Now predicting (test): user ", i)
        prediction = _predict(m, i, sl, neighbors, averages, deviations)
        test_predictions.append(prediction)
        test_targets.append(target)
        i += 1

    return train_predictions, train_targets, test_predictions, test_targets

def calculate_rmse(prediction ,target):
    """
    Calculate mean squared error
    
    Parameters
    --------
    """
    p = np.array(prediction)
    t = np.array(target)
    return np.sqrt(np.mean((p-t)**2))
                
if __name__ == '__main__':
    print("Now: Load data.")
    user2movie, movie2user, usermovie2rating, user2movie_test, movie2user_test, usermovie2rating_test = load_data()
    print("Now: Calculate coefficient values.")
    K = 20
    limit = 10
    sl, neighbors, averages, deviations = calculate_coef(K, limit, user2movie, movie2user, movie2user_test)
    print("Now: Perform Collaborative Filtering.")
    train_predictions, train_targets, test_predictions, test_targets = predict(usermovie2rating, usermovie2rating_test, sl, neighbors, averages, deviations)
    print('Train rmse: ', calculate_rmse(train_predictions, train_targets))
    print('Test rmse: ', calculate_rmse(test_predictions, test_targets))