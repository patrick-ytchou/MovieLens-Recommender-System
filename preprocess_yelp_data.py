import numpy as np
import pandas as pd

state_codes = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
    'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 
    'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
    'VT', 'VA', 'WA', 'WV', 'WI', 'WY'   
]


## Parse Review
print("Now: start parsing review data.")
review_list = []
dtypes = {"stars": np.float16,
         "useful": np.int32,
         "funny": np.int32,
         "cool": np.int32}
chunks = pd.read_json('data/yelp_academic_dataset_review.json'
                     , lines=True
                     , chunksize=1024
                     , orient='records'
                     , dtype = dtypes)

for chunk in chunks:
    chunk = chunk.query("date >= '2018-01-01'")
    review_list.append(chunk)

review = pd.concat(review_list, ignore_index=True)
review.reset_index(drop=True)#.to_csv('data/review.csv')
print("Shape of review: ", review.shape)
unique_business = list(set(review.business_id))



## Parse Business
print("Now: start parsing business data.")
business_list = []
chunks = pd.read_json('data/yelp_academic_dataset_business.json'
                     , lines=True
                     , chunksize=1024
                     , orient='records')

for chunk in chunks:
    chunk = chunk[chunk['business_id'].isin(unique_business)]
    chunk = chunk[chunk['state'].isin(state_codes)]
    business_list.append(chunk)
    
business = pd.concat(business_list, ignore_index=True)
business.reset_index(drop=True).to_csv('data/business.csv')
unique_business_v2 = list(set(business.business_id))
print("Shape of business: ", business.shape)


## update review & unique user list
print("Now: generate review data in USA.")
review = review[review.business_id.isin(unique_business_v2)]
review.to_csv('data/review.csv')
unique_user = list(set(review.user_id))



## Parse User
print("Now: start parsing user data.")
user_list = []
chunks = pd.read_json('data/yelp_academic_dataset_user.json'
                     , lines=True
                     , chunksize=1024
                     , orient='records')

for chunk in chunks:
    chunk = chunk[chunk['user_id'].isin(unique_user)]
    user_list.append(chunk)
    
user = pd.concat(user_list, ignore_index=True)
user.reset_index(drop=True).to_csv('data/user.csv')
print("Shape of user: ", user.shape)