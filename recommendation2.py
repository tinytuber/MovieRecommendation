#%%
import pandas as pd
import numpy as np
#import tensorflow as tf 
#import tensorflow_recommenders as tfrs
#from sklearn.model_selection import train_test_split
from surprise import Dataset, SVD, Reader 
from surprise.model_selection import train_test_split
from collections import defaultdict
# %%
ratings = pd.read_csv("/Users/chchow/Documents/AI Work/MovieRecommendation/ml-latest-small/ratings.csv")
movies = pd.read_csv("/Users/chchow/Documents/AI Work/MovieRecommendation/ml-latest-small/movies.csv")
# %%
reader = Reader(rating_scale = (1,5))
surprise_data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = surprise_data.build_full_trainset()
model = SVD()
model.fit(trainset)
# %%
def get_top_n_recommendations(predictions, n=5):
  top_n = defaultdict(list)
  for uid, iid, true_r, est, _ in predictions:
    top_n[uid].append((iid, est))
  for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = user_ratings[:n]
  return top_n
# %%
new_user_rating = {
    #'movieId':[88125, 164179, 318, 356],
    "movieId":[4896,5816,8368,40815,54001,69844],
    'rating': [1,1,1,1,5,5]
}
new_user_df = pd.DataFrame(new_user_rating)
new_user_id = 200_000_000
# Create a list of (user, item, rating) tuples for the new user
new_ratings_list = [(new_user_id, movie_id, rating) for movie_id, rating in zip(new_user_df['movieId'], new_user_df['rating'])]

# Make predictions for all movies for the new user
predictions = []
for movie_id in ratings['movieId'].unique(): # Predict for movies in the training data
    prediction = model.predict(new_user_id, movie_id)
    predictions.append(prediction)

# %%
# Get top N recommendations for the new user
top_n = get_top_n_recommendations(predictions, n=4)

print(f"Top 5 movie recommendations for user {new_user_id}:")
if new_user_id in top_n:
    for movie_id, rating in top_n[new_user_id]:
        print(f"Movie ID: {movie_id}, Predicted Rating: {rating:.2f}")
else:
    print("No recommendations could be generated for this user.")
# %%
# Combine original ratings with new user ratings
all_ratings = pd.concat([ratings, new_user_df], ignore_index=True)

# Prepare dataset
reader = Reader(rating_scale=(0.5, 5))
surprise_data = Dataset.load_from_df(all_ratings[["userId", "movieId", "rating"]], reader)

# Train on full dataset
trainset = surprise_data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Predict for movies the new user hasn't rated
watched_movies = set(new_user_df['movieId'])
all_movie_ids = set(ratings['movieId'].unique())
unwatched_movies = list(all_movie_ids - watched_movies)

predictions = [model.predict(new_user_id, movie_id) for movie_id in unwatched_movies]

# Get top N recommendations
def get_top_n_recommendations(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n_recommendations(predictions, n=5)

# Print recommendations
print(f"Top 5 movie recommendations for user {new_user_id}:")
for movie_id, rating in top_n[new_user_id]:
    title = movies.loc[movies['movieId'] == movie_id, 'title'].values[0]
    print(f"{title} (Movie ID: {movie_id}) – Predicted Rating: {rating:.2f}")
# %%
