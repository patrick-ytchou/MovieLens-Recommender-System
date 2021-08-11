## Yelp Recommender System
Implementation of Recommender System on MovieLens Dataset

### __Project Introduction__

This is a matrix factorization based recommender system hosted on AWS EC2 instances. 
It aims to provide a interface for user to input its preference to different kinds of movies.
From their initial input, the model will generate recommendations accordingly. 

#### __How to Implement__

From the navgation drop-down selection on the left, choose **Matrix Factorization** and follow the instructure accordingly.

---
### __System Design__
This temporary system structure is illustrated in the image below.
Since the incoming traffic is estimated to be quite low, we simply use a application load balancer with an auto-scaling group.


*Future Improvement*: 

1. Utilize AWS SQS to queue requests before heading directly to auto-scalinbg groups.
2. Substitute EC2 instances with Lambda functions to make it a serverless process.

---
    
### __Model Structure__
The model structure is summarized below.

```python
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 30)        30734520    input_1[0][0]                    
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1, 30)        3479370     input_2[0][0]                    
__________________________________________________________________________________________________
dot (Dot)                       (None, 1, 1)         0           embedding[0][0]                  
                                                                embedding_1[0][0]                
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 1, 1)         1024484     input_1[0][0]                    
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 1, 1)         115979      input_2[0][0]                    
__________________________________________________________________________________________________
add (Add)                       (None, 1, 1)         0           dot[0][0]                        
                                                                embedding_2[0][0]                
                                                                embedding_3[0][0]                
__________________________________________________________________________________________________
flatten (Flatten)               (None, 1)            0           add[0][0]                        
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            2           flatten[0][0]                    
__________________________________________________________________________________________________
add_1 (Add)                     (None, 1)            0           flatten[0][0]                    
                                                                dense_2[0][0]                    
==================================================================================================
Total params: 35,354,355
```

This deep residual matrix factorizaiton consists of two part. 


1. __Main Matrix Factorization branch__: 

This branch is the classic implementation of matrix factorization, with **dim** being the inner dimension.

```python
u_embedding = Embedding(N, dim, embeddings_regularizer=l2(reg))(u) # (N, 1, dim)
m_embedding = Embedding(M, dim, embeddings_regularizer=l2(reg))(m) # (M, 1, dim)

u_bias = Embedding(N, 1, embeddings_regularizer=l2(reg))(u) # (N, 1, 1)
m_bias = Embedding(M, 1, embeddings_regularizer=l2(reg))(m) # (M, 1, 1)
x = Dot(axes=2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x) # (N, 1)
```

2. __Deep Residual branch__:

The deep residual branch is basically a neural network that trains the parameters.

```python
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
```

These two branches are later added up together.

```python
x = Add()([x, y])
model = Model(inputs = [u, m], outputs = x)
model.compile(
    loss = 'mse',
    optimizer = Adam(learning_rate=0.01),
    metrics = ['mse']
)
```

### Repo Structure
