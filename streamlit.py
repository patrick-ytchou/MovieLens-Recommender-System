import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import glob
import random
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dot, Add, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from tqdm import tqdm
from stqdm import stqdm

st.set_page_config(layout="wide")
# st.set_page_config(base="light")


img_list = glob.glob('image/*.png')

movies = pd.read_csv('data/top 150 movies.csv', index_col=0)
# movie_idx_mapping = pd.Series(movies.movie_idx.values,index=movies.title).to_dict()
movie_idx_mapping = pd.Series(movies.movie_idx.values, index=movies.title).to_dict()

@st.cache(allow_output_mutation=True)
def choose_movie(movie_idx_mapping, n):
    return random.sample(list(movie_idx_mapping.keys()), n)

total_movies = 26729
rating = [0] * total_movies

movie_choice = choose_movie(movie_idx_mapping, 5)
movie_idxs = []
for c in movie_choice:
    movie_idxs.append(movie_idx_mapping[c])
# movie_idxs = movie_idx_mapping[movie_choice[0]]
# st.caching.clear_cache()
rerun = st.button("Rerun")
if rerun:
    st.caching.clear_cache()

@st.cache(allow_output_mutation=True)
def create_control():
    control = [False, False, False, False, False]
    user_ratings = [0, 0, 0, 0, 0]
    return control, user_ratings

control, user_ratings = create_control()


col1, col2, col3, col4, col5 = st.beta_columns(5)
with col1:
    movie_idx = movie_idx_mapping[movie_choice[0]]
    image = Image.open(f'image/{movie_choice[0]}.png')
    col1.subheader("Movie #1")
    with st.form("form1"):
        st.image(image, caption=f'Movie: {movie_choice[0]}')
        st.text("")
        slider_val = st.slider("Slider #2", 1, 5, 1)
        submitted = st.form_submit_button("Submit #1")
        if submitted:
            st.write("You've rated movie this movie at: ", slider_val)
            control[0] = True
            user_ratings[0] = slider_val

with col2:
    movie_idx = movie_idx_mapping[movie_choice[1]]
    image = Image.open(f'image/{movie_choice[1]}.png')
    col2.subheader("Movie #2")
    with st.form("form2"):
        st.image(image, caption=f'Movie: {movie_choice[1]}')
        st.text("")
        slider_val = st.slider("Slider #2", 1, 5, 1)
        submitted = st.form_submit_button("Submit #2")
        if submitted:
            st.write("You've rated movie this movie at: ", slider_val)
            control[1] = True
            user_ratings[1] = slider_val

with col3:
    movie_idx = movie_idx_mapping[movie_choice[2]]
    image = Image.open(f'image/{movie_choice[2]}.png')
    col3.subheader("Movie #3")
    with st.form("form3"):
        st.image(image, caption=f'Movie: {movie_choice[2]}')
        st.text("")
        slider_val = st.slider("Slider #3", 1, 5, 1)
        submitted = st.form_submit_button("Submit #3")
        if submitted:
            st.write("You've rated movie this movie at: ", slider_val)
            control[2] = True
            user_ratings[2] = slider_val

with col4:
    movie_idx = movie_idx_mapping[movie_choice[3]]
    image = Image.open(f'image/{movie_choice[3]}.png')
    col4.subheader("Movie #4")
    with st.form("form4"):
        st.image(image, caption=f'Movie: {movie_choice[3]}')
        st.text("")
        slider_val = st.slider("Slider #4", 1, 5, 1)
        submitted = st.form_submit_button("Submit #4")
        if submitted:
            st.write("You've rated movie this movie at: ", slider_val)
            control[3] = True
            user_ratings[3] = slider_val

with col5:
    movie_idx = movie_idx_mapping[movie_choice[4]]
    image = Image.open(f'image/{movie_choice[4]}.png')
    col5.subheader("Movie #5")
    with st.form("form5"):
        st.image(image, caption=f'Movie: {movie_choice[4]}')
        st.text("")
        slider_val = st.slider("Slider #5", 1, 5, 1)
        submitted = st.form_submit_button("Submit #5")
        if submitted:
            st.write("You've rated movie this movie at: ", slider_val)
            control[4] = True
            user_ratings[4] = slider_val

# show_df = pd.DataFrame(
#     [np.array(["movie #1", "movie #2", "movie #3", "movie #4", "movie #5"]),control,user_ratings]
#     ).T
# show_df = ['movie_idx','control', 'user_rating']

dummy_idxs = ["movie #1", "movie #2", "movie #3", "movie #4", "movie #5"]
show_df = pd.DataFrame(
    [movie_choice,control,user_ratings]
    ).T
show_df.columns = ['movie_name','has_rating', 'user_rating']


user_input_df = pd.DataFrame(
    [movie_idxs, control, user_ratings]
    ).T
user_input_df.columns = ['movie_idx', 'control', 'user_rating']
st.dataframe(show_df)

def find_closest_user(user_input_df):
    """
    Look for closest user.
    """
    df = pd.read_csv('data/edited_rating.csv', index_col=0)
    user_ttl_rating = df.groupby(["userId"])[['rating']].count().sort_values(by='rating', ascending=False).reset_index()
    user_ttl_rating.columns = ['userId', 'numRating']
    
    
    
    filtered_df = df[df['movie_idx'].isin(user_input_df.movie_idx.values)][['userId', 'movie_idx', 'rating']]
    filtered_df = pd.merge(filtered_df, user_ttl_rating, on = 'userId')

    concat = pd.merge(filtered_df, user_input_df, on='movie_idx')
    concat['mse'] = (concat['user_rating'] - concat['rating'])**2

    result = concat.groupby(['userId','numRating'])[['mse']].agg(['sum', 'count']).reset_index(drop=False)
    result.columns = ['userId', 'numRating', 'mse', 'count']
    result['avg_mse'] = result['mse'] / result['count']
    result['adjusted_avg_mse'] = result['avg_mse'] / np.sqrt(result['count']) / np.cbrt(result['numRating'])
    
    temp = result[result['adjusted_avg_mse'] == min(result['adjusted_avg_mse'])]
    userid = temp[temp['count'] == max(temp['count'])].userId.values[0]
    return userid

def predict(model, userid):
    """
    Build neural network model.
    """
    user = pd.Series([userid] * 26744)
    movie = pd.Series(list(range(26744)))
    
    result = model.predict(
        x = [user, movie],
        # x = [user_input.userId.values, user_input.movie_idx.values],
        verbose = 1
        )
    result = np.array(result).flatten()
    print(result)
    return result

def modeling(N, M, dim, reg):
    """
    Conduct modeling in Keras.
    """
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
    return model

with st.beta_container():
    if sum(control) != 5:
        st.write("No recommendation generated as you haven't rate all movies.")
    else:
        # pbar = tqdm(total=5)
        pbar = stqdm(total=5)
        
        st.write("Generating Movie Prediction.")
        N = 138493
        M = 26744
        model = modeling(N = N, M = M, dim = 10, reg = 0.01)
        model.load_weights('model/mf_deep_v3_weight.h5')
        pbar.update(1)

        userid = find_closest_user(user_input_df)
        pbar.update(1)
        prediction = predict(model, userid)
        pbar.update(1)
        all_movie = pd.read_csv('data/all_movie.csv')
        pbar.update(1)
        top_10_idx = np.argsort(prediction)[-6:]

        inv_movie_mapping_title = pd.Series(all_movie.title, index = all_movie.movie_idx.values).to_dict()
        inv_movie_mapping_genre = pd.Series(all_movie.genres, index = all_movie.movie_idx.values).to_dict()
        pbar.update(1)
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            movie_idx = inv_movie_mapping_title[top_10_idx[-1]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-1]]}.png')
            st.subheader("Movie #1")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-1]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-1]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-1]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-1]]}')
            st.text("")

        with col2:
            movie_idx = inv_movie_mapping_title[top_10_idx[-2]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-2]]}.png')
            st.subheader("Movie #2")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-2]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-2]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-2]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-2]]}')
            st.text("")

        with col3:
            movie_idx = inv_movie_mapping_title[top_10_idx[-3]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-3]]}.png')
            st.subheader("Movie #3")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-3]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-3]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-3]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-3]]}')
            st.text("")
        pbar.update(1)
        col4, col5, col6 = st.beta_columns(3)
        with col4:
            movie_idx = inv_movie_mapping_title[top_10_idx[-4]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-4]]}.png')
            st.subheader("Movie #4")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-4]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-4]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-4]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-4]]}')
            st.text("")

        with col5:
            movie_idx = inv_movie_mapping_title[top_10_idx[-5]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-5]]}.png')
            st.subheader("Movie #5")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-5]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-5]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-5]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-5]]}')
            st.text("")

        with col6:
            movie_idx = inv_movie_mapping_title[top_10_idx[-6]]
            # image = Image.open(f'image/{inv_movie_mapping[top_10_idx[-5]]}.png')
            st.subheader("Movie #6")
            st.text(f"Name: {inv_movie_mapping_title[top_10_idx[-6]]}")
            st.text(f"Genre: {inv_movie_mapping_genre[top_10_idx[-6]]}")
            st.write(f"Link for [more information](https://www.google.com/search?q={inv_movie_mapping_title[top_10_idx[-6]].replace(' ', '%20')}+site%3Awww.imdb.com)")
            # st.image(image, caption=f'Movie: {inv_movie_mapping[top_10_idx[-5]]}')
            st.text("")
        pbar.update(1)        
        st.balloons()        
        
