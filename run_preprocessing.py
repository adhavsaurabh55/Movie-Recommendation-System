import os
import ast
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download nltk data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Create artifacts dir
os.makedirs('artifacts', exist_ok=True)

# Load data
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('ratings.csv')
movies = movies.merge(credits, on='title')

# Select columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew', 'release_date', 'vote_average']]

# Clean data
movies.dropna(inplace=True)
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
movies.dropna(inplace=True)

# Convert functions
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

# Apply conversions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# New DF
new_df = movies[['movie_id', 'title', 'tags', 'year', 'vote_average']]

# String join + lower
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()
def stems(text):
    T = []
    for i in text.split():
        T.append(ps.stem(i))
    return " ".join(T)
new_df['tags'] = new_df['tags'].apply(stems)

# Vectorize + similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

# Save
pickle.dump(new_df.to_dict(), open('artifacts/movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

print("Artifacts generated successfully: movie_dict.pkl and similarity.pkl")
print(f"Dataset shape: {new_df.shape}")
