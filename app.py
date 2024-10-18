from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from sklearn.impute import KNNImputer

app = Flask(__name__)

# Load the datasets
ratings = pd.read_csv('ratings.csv')  
movies = pd.read_csv('movies.csv')    

# Prepare content-based recommendations using genres and titles
movies['genres'] = movies['genres'].fillna('')  # Handle missing genres
movies['title'] = movies['title'].fillna('')    # Handle missing titles
tfidf_genres = TfidfVectorizer(stop_words='english')  # Initialize TF-IDF Vectorizer for genres
tfidf_titles = TfidfVectorizer(stop_words='english')   # Initialize TF-IDF Vectorizer for titles

tfidf_matrix_genres = tfidf_genres.fit_transform(movies['genres'])  # Convert 'genres' into a matrix of TF-IDF features
tfidf_matrix_titles = tfidf_titles.fit_transform(movies['title'])    # Convert 'title' into a matrix of TF-IDF features

# Calculate cosine similarity matrix for the movies based on genres and titles
content_similarity_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
content_similarity_titles = cosine_similarity(tfidf_matrix_titles, tfidf_matrix_titles)

# Combine genre and title similarities
content_similarity = (content_similarity_genres + content_similarity_titles) / 2

# Function to get movie recommendations based on genre and title similarity (content-based filtering)
def get_content_based_recommendations(movie_title, num_recommendations=15):
    movie_title = movie_title.strip().lower()
    titles = movies['title'].tolist()
    best_match, score = process.extractOne(movie_title, titles)

    if score < 80:  
        return [], None  

    idx = movies[movies['title'] == best_match].index[0]
    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    
    return movies['title'].iloc[movie_indices].tolist(), best_match

# Create user-item matrix for user-based collaborative filtering
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Use KNNImputer to handle missing values (non-rated movies)
imputer = KNNImputer(n_neighbors=5)  # Choose 5 nearest neighbors to impute
user_item_matrix_filled = pd.DataFrame(imputer.fit_transform(user_item_matrix),
                                       index=user_item_matrix.index,
                                       columns=user_item_matrix.columns)

# Calculate user similarity matrix using cosine similarity
user_similarity = cosine_similarity(user_item_matrix_filled)

# Create a DataFrame to store the user similarity scores
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get user-based recommendations
def get_user_based_recommendations(user_id, num_recommendations=10):
    # Get similarity scores for the given user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]

    # Get the top similar user
    top_user = similar_users.index[0]

    # Get movies rated by the similar user but not rated by the target user
    user_rated_movies = user_item_matrix.loc[user_id]
    similar_user_rated_movies = user_item_matrix.loc[top_user]

    # Recommend movies that the similar user has rated highly but the target user hasn't rated
    recommendations = similar_user_rated_movies[(similar_user_rated_movies > 0) & (user_rated_movies.isna())]

    # Sort the recommendations by rating and return the top N recommendations
    recommended_movie_ids = recommendations.sort_values(ascending=False).head(num_recommendations).index
    recommended_movie_titles = movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

    return recommended_movie_titles

# Dictionary to store search history for each user
search_history = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id']
    movie_title = request.form['movie_title']

    # Store the search in the user's history
    if user_id not in search_history:
        search_history[user_id] = []
    if movie_title and movie_title not in search_history[user_id]:
        search_history[user_id].append(movie_title)

    # Get recommendations based on user ID and movie title
    user_based_recommendations = get_user_based_recommendations(int(user_id))
    content_recommendations, searched_movie = [], None

    if movie_title:  # If a movie title is provided
        content_recommendations, searched_movie = get_content_based_recommendations(movie_title)

    # Generate additional content-based recommendations from search history
    additional_content_recommendations = []
    for searched_movie_title in search_history[user_id]:
        additional_recs, _ = get_content_based_recommendations(searched_movie_title)
        additional_content_recommendations.extend(additional_recs)

    # Remove duplicates and limit recommendations
    additional_content_recommendations = list(set(additional_content_recommendations))[:10]

    # Prepare results
    recommendations = {
        'user_based': user_based_recommendations,
        'content_based': content_recommendations,
        'additional_content_based': additional_content_recommendations,
        'searched_movie': {'title': searched_movie} if movie_title else {'title': None},
        'search_history': search_history.get(user_id, [])
    }

    return jsonify(recommendations)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    titles = movies['title'].tolist()
    matches = [title for title in titles if query in title.lower()]

    return jsonify(matches)

@app.route('/history/<user_id>', methods=['GET'])
def get_search_history(user_id):
    history = search_history.get(user_id, [])
    return jsonify({'history': history})

if __name__ == '__main__':
    app.run(debug=True)