#%% 
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# %%
user_ratings_df = pd.read_csv("/Users/chchow/Documents/AI Work/MovieRecommendation/archive (1)/ratings.csv")
user_ratings_df["movieId"] = user_ratings_df["movieId"].astype(int)
user_ratings_df.head()
# %%
movie_metadata = pd.read_csv("/Users/chchow/Documents/AI Work/MovieRecommendation/archive (1)/movies_metadata.csv")
#movie_metadata = movie_metadata[["id", "title", "genres"]]
movie_metadata = movie_metadata.rename(columns={"id": "movieId"})
movie_metadata["movieId"] = pd.to_numeric(movie_metadata["movieId"], errors="coerce")
movie_metadata = movie_metadata.dropna(subset = ["movieId"])
movie_metadata.head()
#%%
credits = pd.read_csv("/Users/chchow/Documents/AI Work/MovieRecommendation/archive (1)/credits.csv")
credits = credits.rename(columns={"id":"movieId"})
# %%
users_and_metadata = pd.merge(user_ratings_df, movie_metadata, how="inner", on="movieId")
credits_and_metadata = pd.merge(credits, movie_metadata, how="inner", on="movieId")
# %%
# Summary statistics
#n_users = len(credits_and_metadata['userId'].unique())
n_movies = len(credits_and_metadata['title'].unique())
# %%
movies = users_and_metadata[["userId", "movieId", "rating", "title"]]
# %%
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from sklearn.model_selection import train_test_split
# %%
user_id = users_and_metadata["userId"].unique().tolist()
movie_id = users_and_metadata["movieId"].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_id)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_id)}

users_and_metadata["user"] = users_and_metadata["userId"].map(user2user_encoded)
users_and_metadata["movie"] = users_and_metadata["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# %%
X = users_and_metadata[["user", "movie"]].values
y = users_and_metadata["rating"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
# %%
embedding_size = 10

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users, embedding_size)(user_input)
movie_embedding = Embedding(num_movies, embedding_size)(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)
dot_product = Dot(axes=1)([user_vec, movie_vec])

model = Model(inputs=[user_input, movie_input], outputs = dot_product)
model.compile(optimizer="adam", loss="mse")

model.summary()
# %%
history = model.fit(
    [X_train[:, 0], X_train[:,1]],
    y_train,
    batch_size = 256,
    epochs = 3,
    validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
    verbose=1
)
# %%
def predict_rating(user_id, movie_id):
    u = user2user_encoded.get(user_id)
    m = movie2movie_encoded.get(movie_id)
    if u is None or m is None:
        return "User or movie not in training data"
    pred = model.predict([[u], [m]], verbose = 0)
    return float(pred[0][0])

print(predict_rating(user_id=1, movie_id=101))