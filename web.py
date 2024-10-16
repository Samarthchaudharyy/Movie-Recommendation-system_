import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load snowfall animation
lottie_snowfall = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_u4yrau.json")

# Load data
movies_data = pd.read_csv('movies.csv')

# Fill missing values and combine selected features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

# Function to recommend movies
def get_recommendations(movie_name, num_recommendations=10):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

    if not find_close_match:
        return ["Movie not found! Please try a different title."]
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
    
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:num_recommendations + 1], 1):  # Skip the first one
        index = movie[0]
        title_from_index = movies_data.iloc[index].title
        recommended_movies.append(f"{i}. {title_from_index}")
    return recommended_movies

# Streamlit App
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥")

# Adding an image to the website
st.image("/content/movie.jpg", use_column_width=True)

st.title("Welcome to the Movie Recommendation System 🎬")
st.write("Enter your favorite movie and get recommendations based on similar movies!")

# Input movie from user
movie_name = st.text_input("Enter your favorite movie:")

# If user enters a movie name
if movie_name:
    recommendations = get_recommendations(movie_name)
    if "Movie not found!" in recommendations:
        st.error("Movie not found! Please try a different title.")
    else:
        st.subheader(f"Movies recommended for '{movie_name}':")
        for movie in recommendations:
            st.write(movie)

        # Show snowfall animation after displaying results
        st_lottie(lottie_snowfall, height=300, width=300)

# Sidebar: About section
st.sidebar.title("About")
st.sidebar.info(
    """
    **Movie Recommendation System** is an AI-based recommendation tool that provides you with movie suggestions based on your input. 
    It analyzes movie features like genres, cast, and director to find movies similar to the one you like.
    
    This web app is built using **Streamlit**. Enjoy movie recommendations with a touch of interactivity and animation!
    """
)

