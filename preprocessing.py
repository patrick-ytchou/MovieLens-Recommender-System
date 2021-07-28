import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.utils import shuffle
from pathlib import Path
import os

def preprocessing(df):
    """
    Proprocess the data accordingly.
    """
    
    # make userid goes from 0...N-1
    df.userId = df.userId - 1
    
    # create a new mapping for movie id to create a continuous ids    
    unique_ids = set(df.movieId.values)
    movie2idx = {}
    new_id = 0
    for old_id in unique_ids:
        movie2idx[old_id] = new_id
        new_id += 1    
    
    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
    df = df.drop(columns=['timestamp'])
    df.to_csv('data/edited_rating.csv')
    
    return df

def shrink(df, n, m):
    """
    Shrink the dataset to 
    
    Parameters
    --------
    n: int, number of users
    m: int, number of movies
    """
    print("original dataframe size: ", len(df))
    
    N = df.userId.max() + 1
    M = df.movieId.max() + 1
    
    ## retrieve the top n users and m movies to shrink the dataset
    user_ids = [u for u, c in Counter(df.userId).most_common(n)]
    movie_ids = [m for m, c in Counter(df.movie_idx).most_common(m)]
    
    df = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)]
    
    ## remake sequential user ids and movie ids
    movie2idx = {}
    nmid = 0
    for omid in movie_ids:
        movie2idx[omid] = nmid
        nmid += 1
    
    user2idx = {}
    nuid = 0
    for ouid in user_ids:
        user2idx[ouid] = nuid
        nuid += 1
    
    
    ## set new ids
    # df.loc[:, 'movie_idx'] = df.apply(lambda row: movie2idx[row.movie_idx], axis=1)
    # df.loc[:, 'userId'] = df.apply(lambda row: user2idx[row.userId], axis=1)
    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movie_idx], axis=1)
    df['userId'] = df.apply(lambda row: user2idx[row.userId], axis=1)
    
    df.to_csv('data/shrinked_rating.csv')
    return df

def create_user_movie_rating_dict(df, train_size = 0.8):
    """
    Create sparse table for user, movie, and rating.
    
    Parameters
    --------
    train_size: float, size of the training data set
    """
    
    if os.path.exists("data/dict"):
        ## If exist, no need to run again
        return 
    else:
        N = df.userId.max() + 1
        M = df.movie_idx.max() + 1
        
        ## Create training and testing set
        df = shuffle(df)
        cutoff_point = int(train_size * len(df))
        df_train = df.iloc[:cutoff_point]
        df_test = df.iloc[cutoff_point:]
        
        ## create user2movie & movie2user hash table for train data
        user2movie, movie2user, usermovie2rating = {}, {}, {}
        
        print("Now: Creating user_movie_rating for train data.")
        
        def update_rating(row):
            i = int(row.userId)
            j = int(row.movie_idx)
            
            ## update user2movie table
            if i not in user2movie:
                user2movie[i] = [j]
            else:
                user2movie[i].append(j)
            
            ## update movie2user table
            if j not in movie2user:
                movie2user[j] = [i]
            else:
                movie2user[j].append(i)
            
            ## update usermovie2rating table
            usermovie2rating[(i, j)] = row.rating
        
        df_train.apply(update_rating, axis=1)
        
        
        ## create user2movie & movie2user hash table for test data
        user2movie_test, movie2user_test, usermovie2rating_test = {}, {}, {}
        
        print("Now: Creating user_movie_rating for test data.")
        
        def update_rating_test(row):
            i = int(row.userId)
            j = int(row.movie_idx)
            
            ## update user2movie table
            if i not in user2movie_test:
                user2movie_test[i] = [j]
            else:
                user2movie_test[i].append(j)
            
            ## update movie2user table
            if j not in movie2user_test:
                movie2user_test[j] = [i]
            else:
                movie2user_test[j].append(i)
            
            ## update usermovie2rating table
            usermovie2rating_test[(i, j)] = row.rating
        
        df_test.apply(update_rating_test, axis=1)
        
        # Save pre-computed dictionary
        # filepath = ['data/dict/user2movie.json',
        #             'data/dict/movie2user.json',
        #             'data/dict/usermovie2rating.json',
        #             'data/dict/user2movie_test.json',
        #             'data/dict/movie2user_test.json',
        #             'data/dict/usermovie2rating_test.json']
        
        # for i in range(len(filepath)):
        #     var_name = filepath[i].split('/')[1].split('.')[0]
        #     with open(filepath[i], 'wb') as f:
        #         pickle.dump(eval(var_name), f)

        with open('data/user2movie.json', 'wb') as f:
            pickle.dump(user2movie, f)    
        with open('data/movie2user.json', 'wb') as f:
            pickle.dump(movie2user, f)    
        with open('data/usermovie2rating.json', 'wb') as f:
            pickle.dump(usermovie2rating, f)    
        with open('data/user2movie_test.json', 'wb') as f:
            pickle.dump(user2movie_test, f)    
        with open('data/movie2user_test.json', 'wb') as f:
            pickle.dump(movie2user_test, f)    
        with open('data/usermovie2rating_test.json', 'wb') as f:
            pickle.dump(usermovie2rating_test, f)    


if __name__ == '__main__':
    print("Now: Reading Raw Data.")
    df = pd.read_csv('data/rating.csv')
    print("Now: Preprocessing data.")
    df = preprocessing(df)
    print("Now: Create shrinked data.")
    df = shrink(df, 10000, 2000)
    print("Now: Computing user, movie, rating table.")
    create_user_movie_rating_dict(df, train_size=0.8)
    print("Success: Complete Processing.")