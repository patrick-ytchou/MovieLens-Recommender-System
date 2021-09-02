import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import glob
# import random
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Dot, Add, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import requests

def app():

    rerun = st.button("Rerun")
    if rerun:
        st.caching.clear_cache()
        
    st.title("Yelp Recommmender System")
    text = """
    ---
    
    ## Step 1
    Please rate the following 6 shops accordingly. You can check out it's yelp link for more information
    """
    st.markdown(text)
    business = pd.read_csv('data/top_businesses.csv')
    business_idx_mapping = business.set_index('business_idx').to_dict(orient='index')

    @st.cache(allow_output_mutation=True)
    def choose_business(business_idx_mapping, n):
        return np.random.choice(list(business_idx_mapping.keys()),n, replace=False)


    total_businesses = 115979
    rating = [0] * total_businesses

    business_idxs = choose_business(business_idx_mapping, 6)


    @st.cache(allow_output_mutation=True)
    def create_control():
        control = [False, False, False, False, False, False]
        user_ratings = [0, 0, 0, 0, 0, 0]
        return control, user_ratings

    control, user_ratings = create_control()


    # Set requests
    key = 'qH6lx8n0QgpbkxCSeoBugx0OaSUlaxM-KRrJQjqB7moVuGB5TZzCGVdewYhghcNtVt5Mqkofd1xGB9xvgmYp5L0ae-yNIbBqhgark3I29rDFrlwNaqaJqwwDqOQHYXYx'
    headers = {'Authorization': f'Bearer {key}'}

    business_names = []

    @st.cache(allow_output_mutation=True)
    def get_api_responses(business_idxs):
        responses = []
        for bid in business_idxs:
            url = f"https://api.yelp.com/v3/businesses/{business_idx_mapping[bid]['business_id']}"
            response = requests.get(url, headers = headers).json()
            responses.append(response)
        return responses

    # business_infos, business_ids, responses = get_api_responses(business_idxs)
    responses = get_api_responses(business_idxs)

    
    # col1, col2, col3, col4, col5 = st.beta_columns(5)
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        response = responses[0]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col1.subheader("Store #1")
        with st.form("form1"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #1", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[0] = True
                user_ratings[0] = slider_val

    with col2:
        response = responses[1]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col2.subheader("Store #2")
        with st.form("form2"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #2", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[1] = True
                user_ratings[1] = slider_val

    with col3:
        response = responses[2]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col3.subheader("Store #3")
        with st.form("form3"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #3", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[2] = True
                user_ratings[2] = slider_val

    col4, col5, col6 = st.beta_columns(3)
    with col4:
        response = responses[3]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col4.subheader("Store #4")
        with st.form("form4"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #4", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[3] = True
                user_ratings[3] = slider_val

    with col5:
        response = responses[4]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col5.subheader("Store #5")
        with st.form("form5"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #5", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[4] = True
                user_ratings[4] = slider_val

    with col6:
        response = responses[5]
        
        image_url = response['image_url']
        image = Image.open(requests.get(image_url, stream=True).raw)
        col6.subheader("Store #6")
        with st.form("form6"):
            st.markdown(f"**Name**: {response['name']}")
            business_names.append(response['name'])
            st.image(image, caption=f"Business: {response['name']}")
            st.markdown(f"**Address**: {', '.join(response['location']['display_address'])}")
            categories = []
            for cat in response['categories']:
                categories.append(cat['title'])
            st.markdown(f"**Categories**: {', '.join(categories)}")
            st.markdown(f"[_Yelp link_]({response['url']})")
            slider_val = st.slider("Rating #6", 1, 5, 1)
            submitted = st.form_submit_button("Submit your rating")
            if submitted:
                st.write("You've rated at: ", slider_val)
                control[5] = True
                user_ratings[5] = slider_val


    dummy_idxs = ["business #1", "business #2", "business #3", "business #4", "business #5"]
    show_df = pd.DataFrame(
        [business_names, control, user_ratings]
        ).T
    show_df.columns = ['business_name','has_rating', 'user_rating']


    user_input_df = pd.DataFrame(
        [business_idxs, control, user_ratings]
        ).T
    user_input_df.columns = ['business_idx', 'control', 'user_rating']
    st.dataframe(show_df)

    def find_closest_user(user_input_df):
        """
        Look for closest user.
        """
        df = pd.read_csv('data/edited_review.csv', index_col=0)
        business_ttl_review = df.groupby(['user_idx','business_idx','business_id'])[['stars']].count().sort_values(by='stars', ascending=False).reset_index()
        business_ttl_review.columns = ['user_idx','business_idx','business_id', 'numReview']
        
        
        
        filtered_df = df[df['business_idx'].isin(user_input_df.business_idx.values)][['user_idx', 'business_idx', 'stars']]
        filtered_df = pd.merge(filtered_df, business_ttl_review, on = ['user_idx','business_idx'])
        concat = pd.merge(filtered_df, user_input_df, on='business_idx')
        concat['mse'] = (concat['user_rating'] - concat['stars'])**2

        result = concat.groupby(['user_idx','numReview'])[['mse']].agg(['sum', 'count']).reset_index(drop=False)
        result.columns = ['user_idx', 'numReview', 'mse', 'count']
        result['avg_mse'] = result['mse'] / result['count']
        result['adjusted_avg_mse'] = result['avg_mse'] / np.sqrt(result['count']) / np.cbrt(result['numReview'])
        
        temp = result[result['adjusted_avg_mse'] == min(result['adjusted_avg_mse'])]
        userid = temp[temp['count'] == max(temp['count'])].user_idx.values[0]
        return userid

    def predict(model, userid, total_businesses):
        """
        Build neural network model.
        
        Parameters:
        --------
        model: 
        userid:
        total_businesses: int, number of businesses
        """
        user = pd.Series([userid] * total_businesses)
        business = pd.Series(list(range(total_businesses)))
        
        result = model.predict(
            x = [user, business],
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
        if sum(control) != 6:
            st.write("No recommendation generated as you haven't rate all businesses.")
        else:
            
            st.write("Now: Loading model information")
            N = 1024484
            M = 115979
            model = modeling(N = N, M = M, dim = 30, reg = 0.1)
            model.load_weights('model/model_weight_v1.h5')

            userid = find_closest_user(user_input_df)
            st.write("Now: Making prediction")
            prediction = predict(model, userid, total_businesses = M)
            all_business = pd.read_csv('data/all_business.csv')
            top_6_idx = np.argsort(prediction)[-6:]

            business_mapping_id = pd.Series(all_business.business_id, index = all_business.business_idx.values).to_dict()
            
            col1, col2, col3 = st.beta_columns(3)
            with col1:
                business_id = business_mapping_id[top_6_idx[-1]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)
                st.subheader("Business #1")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")

            with col2:
                business_id = business_mapping_id[top_6_idx[-2]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)

                st.subheader("Business #2")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")

            with col3:
                business_id = business_mapping_id[top_6_idx[-3]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)

                st.subheader("Business #3")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")

            col4, col5, col6 = st.beta_columns(3)
            with col4:
                business_id = business_mapping_id[top_6_idx[-4]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)
                st.subheader("Business #4")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")

            with col5:
                business_id = business_mapping_id[top_6_idx[-5]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)

                st.subheader("Business #5")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")

            with col6:
                business_id = business_mapping_id[top_6_idx[-6]]
                url = f"https://api.yelp.com/v3/businesses/{business_id}"
                response = requests.get(url, headers = headers).json()
                image_url = response['image_url']
                image = Image.open(requests.get(image_url, stream=True).raw)

                st.subheader("Business #6")
                st.text(f"Name: {response['name']}")
                st.image(image, caption=f"Business: {response['name']}")
                st.text(f"Address: {', '.join(response['location']['display_address'])}")
                st.text(f"Rating: {response['rating']}")
                st.text(f"Phone: {response['phone']}")
                st.text(f"Is Open Now: {response['hours'][0]['is_open_now']}")
                categories = []
                for cat in response['categories']:
                    categories.append(cat['title'])
                st.text(f"Categories: {', '.join(categories)}")
                st.write(f"Yelp link for [more information]({response['url']})")
                st.text("")     
            st.balloons()        
            
