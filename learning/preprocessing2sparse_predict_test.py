import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

def load_data(filepath):
    """
    Load in th data.
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



def create_sparse_data(input_filepath, output_filepath, train_size):
    """
    Create sparse matrix.
    lil_matrix is better for adding values, and csr_matrix is better for saving.
    """
    print("Now: Start creating sparse data.")
    df = load_data(input_filepath)
    N, M, df_train, df_test = preprocessing(df, 0.8)
    
    
    A = lil_matrix((N,M))
    global count  
    count = 0 
    
    def update(row):
        global count  
        count += 1
        if count % 100000 == 0:
            print("Processed: %.3f" % (float(count)/len(df)))
        
        i = int(row.userId)
        j = int(row.movie_idx)
        A[i, j] = row.rating
    
    df.apply(update, axis = 1)
    
    A = A.tocsr()
    mask = (A>0)
    save_npz(output_filepath, A)

if __name__ == '__main__':
    create_sparse_data(input_filepath = 'data/sample_input.csv', 
                       output_filepath = 'data/sample_input.npz',
                       train_size = 0.8)