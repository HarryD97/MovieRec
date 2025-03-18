import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import re
import requests
import datetime
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from db_connect import get_db_connection

# Import database query functions
from db_query import (
    query_movie_details,
    query_user_ratings,
    query_all_movies,
    query_all_ratings
)

# OMDb API configuration
OMDB_API_KEY = "85f65999"  # Replace with your OMDb API key

# Function to get movie IMDb ID from database
def get_movie_imdb_id(movie_id):
    """Get the IMDb ID for a specific movie"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        # Convert movie_id to integer
        movie_id_int = int(float(movie_id))
        
        cursor = conn.cursor()
        cursor.execute(
            'SELECT l."imdbId" FROM links l WHERE l."movieId" = %s',
            (movie_id_int,)
        )
        result = cursor.fetchone()
        
        if result and result[0]:
            # Format IMDb ID to have 7 digits with leading zeros
            imdb_id = f"tt{int(result[0]):07d}"
            return imdb_id
        return None
    except Exception as e:
        st.error(f"Error fetching IMDb ID: {str(e)}")
        return None
    finally:
        conn.close()

# Function to get movie details from OMDb API
def get_movie_details_from_omdb(imdb_id):
    """Fetch movie details from OMDb API using IMDb ID"""
    if not imdb_id:
        return None
    
    try:
        url = f"http://www.omdbapi.com/?i={imdb_id}&plot=full&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                return data
        return None
    except Exception as e:
        st.error(f"Error fetching from OMDb API: {str(e)}")
        return None

# Function to get detailed movie information combining our database and OMDb
def get_movie_info(movie_id):
    """Get comprehensive movie information from our database and OMDb API"""
    conn = get_db_connection()
    if conn is None:
        return None, None
    
    try:
        # Convert movie_id to integer
        movie_id_int = int(float(movie_id))
        
        # Get basic movie info from our database
        query = """
            SELECT m."movieId", m.title, m.genres, 
                   AVG(r.rating) as avg_rating, COUNT(r.rating) as rating_count
            FROM movies m
            LEFT JOIN ratings r ON m."movieId" = r."movieId"
            WHERE m."movieId" = %s
            GROUP BY m."movieId", m.title, m.genres
        """
        
        movie_info = pd.read_sql_query(query, conn, params=(movie_id_int,))
        
        if movie_info.empty:
            return None, None
        
        # Get IMDb ID
        imdb_id = get_movie_imdb_id(movie_id)
        
        # Get details from OMDb API
        omdb_info = None
        if imdb_id:
            omdb_info = get_movie_details_from_omdb(imdb_id)
        
        return movie_info.iloc[0], omdb_info
    except Exception as e:
        st.error(f"Error fetching movie details: {str(e)}")
        return None, None
    finally:
        conn.close()

# Add new function to add or update user ratings
def add_or_update_rating(user_id, movie_id, rating):
    """
    Add a new rating or update existing rating in the database
    
    Parameters:
    user_id (int): User ID
    movie_id (int): Movie ID
    rating (float): Rating value (0.5-5.0)
    
    Returns:
    bool: True if successful, False otherwise
    str: Success or error message
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "Database connection failed"
        
        cursor = conn.cursor()
        
        # Check if rating already exists
        cursor.execute(
            'SELECT * FROM ratings WHERE "userId" = %s AND "movieId" = %s',
            (user_id, movie_id)
        )
        
        existing_rating = cursor.fetchone()
        timestamp = int(datetime.datetime.now().timestamp())
        
        if existing_rating:
            # Update existing rating
            cursor.execute(
                'UPDATE ratings SET rating = %s, timestamp = %s WHERE "userId" = %s AND "movieId" = %s',
                (rating, timestamp, user_id, movie_id)
            )
            message = f"Rating updated for movie ID {movie_id}"
        else:
            # Insert new rating
            cursor.execute(
                'INSERT INTO ratings ("userId", "movieId", rating, timestamp) VALUES (%s, %s, %s, %s)',
                (user_id, movie_id, rating, timestamp)
            )
            message = f"New rating added for movie ID {movie_id}"
        
        conn.commit()
        conn.close()
        return True, message
    except Exception as e:
        return False, f"Error: {str(e)}"

# Cache database query results
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_movies():
    """Get all movie data and cache results"""
    movies_df = query_all_movies()
    if movies_df.empty:
        st.error("Unable to load movie information from database")
    return movies_df

# Get list of movie genres
@st.cache_data(ttl=3600)
def get_movie_genres():
    """Extract all movie genres from movie data"""
    movies_df = get_movies()
    if movies_df.empty:
        return ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi"]  # Default genres
    
    all_genres = set()
    for genres in movies_df['genres'].str.split('|'):
        if isinstance(genres, list):
            all_genres.update(genres)
    return sorted(list(all_genres))

# Load model function
@st.cache_resource
def load_recommendation_model(model_path=None, model_type="svd", sim_method="cosine"):
    """Load recommendation model, try to find available model if specified path doesn't exist"""
    from user_based_new import EnhancedCF
    
    if model_path is None:
        # Look for model with specified type and similarity method
        default_path = f"model/{model_type}_{sim_method}_model.pkl"
        if os.path.exists(default_path):
            model_path = default_path
        else:
            # Look for any available model
            model_files = []
            for mt in ["svd", "cf"]:
                for sm in ["cosine", "pearson", "manhattan"]:
                    file_path = f"model/{mt}_{sm}_model.pkl"
                    if os.path.exists(file_path):
                        model_files.append(file_path)
            
            if model_files:
                model_path = model_files[0]
                st.info(f"Specified model not found, loading: {model_path}")
            else:
                st.error("No model files found. Please train a model first.")
                return None
    
    # Create and load model
    model = EnhancedCF()
    if model.load_model(filepath=model_path):
        return model
    else:
        st.error(f"Model loading failed: {model_path}")
        return None

# Get all available model files
def get_available_models():
    models = []
    if not os.path.exists("model"):
        return models
        
    for file in os.listdir("model"):
        if file.endswith("_model.pkl"):
            models.append(file)
    return models

# Create user genre heatmap
def create_user_genre_heatmap(user_id):
    """Create rating heatmap for different movie genres by user"""
    # Get user ratings
    user_ratings = query_user_ratings(user_id)
    if user_ratings is None or user_ratings.empty:
        return None
    
    # Get movie information
    movies_df = get_movies()
    if movies_df.empty:
        return None
        
    # Merge ratings with movie data
    user_data = pd.merge(user_ratings, movies_df, on='movieId')
    
    if user_data.empty:
        return None
    
    # Expand movie genres and calculate average rating for each genre
    genre_ratings = []
    for _, row in user_data.iterrows():
        genres = row['genres'].split('|')
        for genre in genres:
            genre_ratings.append({
                'genre': genre,
                'rating': row['rating']
            })
    
    if not genre_ratings:
        return None
        
    genre_df = pd.DataFrame(genre_ratings)
    genre_avg = genre_df.groupby('genre')['rating'].agg(['mean', 'count']).reset_index()
    genre_avg = genre_avg.sort_values('count', ascending=False)
    
    # Create heatmap
    if len(genre_avg) > 0:
        fig = px.bar(
            genre_avg, 
            x='genre', 
            y='mean',
            color='mean',
            labels={'mean': 'Average Rating', 'genre': 'Movie Genre', 'count': 'Number of Ratings'},
            title=f'User {user_id} Genre Preferences',
            hover_data=['count'],
            color_continuous_scale='RdBu_r'
        )
        return fig
    return None

# Get popular movies
@st.cache_data(ttl=3600)
def get_popular_movies(min_ratings=50, top_n=10):
    """Get popular movies with high ratings and many reviews"""
    try:
        # Use SQL query to get popular movies directly
        from db_connect import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
            
        query = f"""
        SELECT r.\"movieId\", COUNT(r.rating) as rating_count, AVG(r.rating) as avg_rating 
        FROM ratings r 
        GROUP BY r.\"movieId\" 
        HAVING COUNT(r.rating) >= {min_ratings} 
        ORDER BY avg_rating DESC, rating_count DESC 
        LIMIT {top_n};
        """
        
        popular = pd.read_sql_query(query, conn)
        
        # Get movie details
        if not popular.empty:
            movies_df = get_movies()
            popular_with_details = pd.merge(
                popular, 
                movies_df, 
                on='movieId', 
                how='inner'
            )
            return popular_with_details
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting popular movies: {e}")
        return pd.DataFrame()

# Create rating widget for movies
def create_rating_widget(movie_id, current_rating=None):
    """Create a rating widget for a movie, with current rating if available"""
    rating_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Set default index based on current rating
    if current_rating is not None:
        # Find the closest rating option
        default_idx = min(range(len(rating_options)), 
                          key=lambda i: abs(rating_options[i] - current_rating))
    else:
        default_idx = 5  # Default to 3.0
    
    return st.select_slider(
        "Your Rating",
        options=rating_options,
        value=rating_options[default_idx],
        format_func=lambda x: f"{'★' * int(x)}{'½' if x % 1 else ''} ({x})"
    )

# Display movie detail page with OMDb details
def show_movie_detail_page(movie_id, user_id):
    """Show detailed movie page with OMDb API data and rating options"""
    # Get movie info from our database and OMDb
    movie_info, omdb_info = get_movie_info(movie_id)
    
    if movie_info is None:
        st.error("Could not retrieve movie information")
        return
    
    # Get user's current rating for this movie if it exists
    user_ratings_df = query_user_ratings(user_id)
    current_rating = None
    
    if user_ratings_df is not None and not user_ratings_df.empty:
        user_movie_rating = user_ratings_df[user_ratings_df['movieId'] == float(movie_id)]
        if not user_movie_rating.empty:
            current_rating = user_movie_rating.iloc[0]['rating']
    
    # Add custom CSS for movie detail page
    st.markdown("""
    <style>
    .movie-poster {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .movie-metadata {
        color: #666;
        font-size: 0.9em;
    }
    .rating-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .imdb-rating {
        background-color: #f5c518;
        color: black;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
        # Back button
    if st.button("Back to Movies", key="back_button"):
        st.session_state.show_movie_detail = False
        st.session_state.selected_movie_id = None
        st.rerun()
    st.markdown("---")
    # Create layout
    if omdb_info and omdb_info.get("Poster") != "N/A":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Show movie poster with a fixed width to make it smaller
            st.image(omdb_info.get("Poster"), caption="", width=400)
    else:
        col1, col2 = st.columns([0, 1])
        col2 = st
    
    with col2:
        # Movie title and year
        if omdb_info:
            st.title(f"{omdb_info.get('Title')} ({omdb_info.get('Year')})")
        else:
            st.title(movie_info['title'])
        
        # Basic information row
        if omdb_info:
            info_cols = st.columns(3)
            
            with info_cols[0]:
                st.markdown(f"**Runtime:** {omdb_info.get('Runtime', 'N/A')}")
            with info_cols[1]:
                st.markdown(f"**Rated:** {omdb_info.get('Rated', 'N/A')}")
            with info_cols[2]:
                imdb_rating = omdb_info.get('imdbRating', 'N/A')
                imdb_votes = omdb_info.get('imdbVotes', 'N/A')
                st.markdown(f"**IMDb:** <span class='imdb-rating'>{imdb_rating}</span> ({imdb_votes} votes)", unsafe_allow_html=True)
        
        # Genres as tags
        if omdb_info and omdb_info.get('Genre'):
            genres = omdb_info.get('Genre').split(', ')
        else:
            genres = movie_info['genres'].split('|')
        
        st.write("**Genres:**")
        genre_html = " ".join([f'<span style="background-color:#e0e0e0; padding:5px; border-radius:5px; margin-right:5px;">{genre}</span>' for genre in genres])
        st.markdown(genre_html, unsafe_allow_html=True)
        
        # Plot/Summary
        st.subheader("Plot")
        if omdb_info and omdb_info.get('Plot') != 'N/A':
            st.write(omdb_info.get('Plot'))
        else:
            st.write("No plot summary available.")
        
        # Cast and crew
        if omdb_info:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Director:**", omdb_info.get('Director', 'N/A'))
                st.write("**Writer:**", omdb_info.get('Writer', 'N/A'))
            with col2:
                st.write("**Actors:**", omdb_info.get('Actors', 'N/A'))
        
        # Ratings from different sources
        if omdb_info and omdb_info.get('Ratings'):
            st.subheader("Ratings")
            for rating in omdb_info.get('Ratings', []):
                st.write(f"**{rating.get('Source')}:** {rating.get('Value')}")
        
        # System rating information
        avg_rating = movie_info['avg_rating'] if 'avg_rating' in movie_info and not pd.isna(movie_info['avg_rating']) else 0
        rating_count = movie_info['rating_count'] if 'rating_count' in movie_info else 0
        st.write(f"**Our System:** {avg_rating:.2f}/5.0 ({rating_count:,} ratings)")
        
        # Add a divider
        st.markdown("---")
        
        # Link to IMDb
        if omdb_info:
            imdb_id = omdb_info.get('imdbID')
            if imdb_id:
                imdb_url = f"https://www.imdb.com/title/{imdb_id}"
                st.markdown(f"[View on IMDb]({imdb_url}) ↗")
    
    # Rating section in a separate container
    st.markdown("---")
    st.subheader("Rate This Movie")
    
    rating_cols = st.columns([3, 2])
    
    with rating_cols[0]:
        # Display current rating
        if current_rating is not None:
            st.write(f"Your current rating: **{current_rating}/5.0**")
            st.write(f"{'★' * int(current_rating)}{'½' if current_rating % 1 else ''}")
        else:
            st.write("You haven't rated this movie yet.")
        
        # Rating widget
        user_rating = create_rating_widget(movie_id, current_rating)
        
        # Submit button
        if st.button("Submit Rating"):
            try:
                user_id_int = int(float(user_id))
                movie_id_int = int(float(movie_id))
                
                success, message = add_or_update_rating(
                    user_id_int,
                    movie_id_int,
                    user_rating
                )
                
                if success:
                    st.success(message)
                    # Update current_rating
                    current_rating = user_rating
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error submitting rating: {str(e)}")
    
    with rating_cols[1]:
        # Display user activity with this movie
        if current_rating is not None:
            st.subheader("Your Activity")
            
            # Fetch user-specific info about this movie
            conn = get_db_connection()
            if conn is not None:
                try:
                    query = """
                        SELECT timestamp 
                        FROM ratings 
                        WHERE "userId" = %s AND "movieId" = %s
                    """
                    user_activity = pd.read_sql_query(query, conn, params=(int(float(user_id)), int(float(movie_id))))
                    conn.close()
                    
                    if not user_activity.empty:
                        # Convert timestamp to readable date
                        timestamp = user_activity.iloc[0]['timestamp']
                        date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%B %d, %Y')
                        st.write(f"You rated this movie on: {date_str}")
                except Exception as e:
                    pass
    


# Main application
def main():
    # Initialize session state for ratings and movie detail page
    if 'rating_status' not in st.session_state:
        st.session_state.rating_status = None
    
    if 'rating_message' not in st.session_state:
        st.session_state.rating_message = ""
    
    if 'show_movie_detail' not in st.session_state:
        st.session_state.show_movie_detail = False
    
    if 'selected_movie_id' not in st.session_state:
        st.session_state.selected_movie_id = None
    
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application title and style
    st.title("🎬 Smart Movie Recommendation System")
    st.markdown("---")
    
    # Sidebar settings
    with st.sidebar:
        st.header("User Settings")
        user_id = st.text_input("Enter User ID", value="1")
        user_id = str(float(user_id))
        # Clear rating status when user ID changes
        if 'last_user_id' not in st.session_state or st.session_state.last_user_id != user_id:
            st.session_state.rating_status = None
            st.session_state.rating_message = ""
            st.session_state.last_user_id = user_id
        
        # Model settings section
        st.header("Model Settings")
        
        # Model selection
        available_models = get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Select Recommendation Model",
                options=available_models,
                index=0
            )
            model_path = os.path.join("model", selected_model)
        else:
            st.warning("No model files found")
            model_path = None
            selected_model = "No models available"
        
        # Parse model type and similarity method
        if selected_model != "No models available":
            model_parts = selected_model.replace("_model.pkl", "").split("_")
            model_type = model_parts[0] if len(model_parts) > 0 else "unknown"
            sim_method = model_parts[1] if len(model_parts) > 1 else "unknown"
            
            st.info(f"Model Type: {model_type.upper()}")
            st.info(f"Similarity Method: {sim_method.capitalize()}")
        
        # Recommendation parameter adjustment
        st.header("Recommendation Parameters")
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
        
        # Similar users adjustment
        n_similar_users = st.slider(
            "Number of Similar Users",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )
        
        # Update settings button
        update_settings = st.button("Update Settings")
    
    # Check if we should show movie detail page
    if st.session_state.show_movie_detail and st.session_state.selected_movie_id is not None:
        show_movie_detail_page(st.session_state.selected_movie_id, user_id)
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Personalized Recommendations",
        "👤 User Profile",
        "🔍 Movie Explorer",
        "⚙️ System Performance"
    ])
    
    # Load model
    with st.spinner("Loading recommendation model..."):
        model = load_recommendation_model(model_path=model_path)
    
    # Tab 1: Personalized Recommendations
    with tab1:
        st.header(f"Personalized Movie Recommendations for User {user_id}")
        
        # Show rating status message if exists
        if st.session_state.rating_status is not None:
            if st.session_state.rating_status:
                st.success(st.session_state.rating_message)
            else:
                st.error(st.session_state.rating_message)
            
            # Add button to clear message
            if st.button("Clear Message"):
                st.session_state.rating_status = None
                st.session_state.rating_message = ""
                st.rerun()
        
        if model is None:
            st.error("Unable to provide recommendations, model loading failed.")
        else:
            # Get user's existing ratings
            user_ratings_df = query_user_ratings(user_id)
            user_ratings = {}
            if user_ratings_df is not None and not user_ratings_df.empty:
                for _, row in user_ratings_df.iterrows():
                    user_ratings[str(row['movieId'])] = row['rating']
            
            # Check if user is in training set
            if user_id in model.trainSet:
                # Get recommendations
                with st.spinner("Generating recommendations..."):
                    # Update model parameters
                    if update_settings:
                        model.n_rec_movie = n_recommendations
                        model.n_sim_user = n_similar_users
                    
                    # Get recommendation results
                    recs = model.recommend(user_id)
                    
                if recs:
                    # Create recommendation data table
                    rec_data = []
                    for movie_id, score in recs:
                        movie_details = query_movie_details(movie_id)
                        if movie_details is not None and not movie_details.empty:
                            title = movie_details.iloc[0]['title']
                            genres = movie_details.iloc[0]['genres']
                            
                            # Check if user has already rated this movie
                            current_rating = user_ratings.get(str(movie_id), None)
                            
                            # Build data row
                            rec_data.append({
                                "Movie ID": movie_id,
                                "Movie Title": title,
                                "Genres": genres,
                                "Score": round(score, 2),
                                "Current Rating": current_rating
                            })
                    
                    # Display recommendation table
                    if rec_data:
                        rec_df = pd.DataFrame(rec_data)
                        
                        # Add filtering options
                        genre_filter = st.multiselect(
                            "Filter by Genre",
                            options=get_movie_genres(),
                            default=[]
                        )
                        
                        # Apply filters
                        filtered_df = rec_df
                        if genre_filter:
                            filtered_df = rec_df[rec_df["Genres"].apply(
                                lambda x: any(genre in x for genre in genre_filter)
                            )]
                        
                        # Display results with clickable movie titles
                        if not filtered_df.empty:
                            # Create custom dataframe with clickable titles
                            for index, row in filtered_df.iterrows():
                                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                                
                                with col1:
                                    # Make title clickable
                                    movie_id = row["Movie ID"]
                                    if st.button(f"{row['Movie Title']}", key=f"rec_{movie_id}"):
                                        st.session_state.show_movie_detail = True
                                        st.session_state.selected_movie_id = movie_id
                                        st.rerun()
                                
                                with col2:
                                    st.write(row["Genres"])
                                
                                with col3:
                                    st.write(f"Score: {row['Score']:.2f}")
                                
                                with col4:
                                    if row["Current Rating"] is not None:
                                        st.write(f"Your Rating: {row['Current Rating']}")
                                    else:
                                        st.write("Not rated")
                            
                            # Display visualization
                            st.subheader("Genre Distribution of Recommended Movies")
                            genres_list = []
                            for genre_str in rec_df["Genres"]:
                                genres_list.extend(genre_str.split("|"))
                            
                            genre_counts = pd.Series(genres_list).value_counts()
                            fig = px.pie(
                                values=genre_counts.values,
                                names=genre_counts.index,
                                title="Genre Distribution of Recommended Movies"
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("No recommended movies match the selected filters.")
                    else:
                        st.warning("Unable to retrieve detailed information for recommended movies.")
                else:
                    st.warning(f"Unable to provide recommendations for user {user_id}.")
            else:
                st.error(f"User {user_id} is not in the training dataset, cannot provide personalized recommendations.")
                
                # Provide cold start solution
                st.subheader("New User? Start with Popular Movies")
                
                # Display popular movies with clickable titles
                popular_movies = get_popular_movies(min_ratings=50, top_n=10)
                if not popular_movies.empty:
                    # Display popular movies
                    for index, row in popular_movies.iterrows():
                        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                        
                        with col1:
                            # Make title clickable
                            movie_id = row["movieId"]
                            if st.button(f"{row['title']}", key=f"pop_{movie_id}"):
                                st.session_state.show_movie_detail = True
                                st.session_state.selected_movie_id = movie_id
                                st.rerun()
                        
                        with col2:
                            st.write(row["genres"])
                        
                        with col3:
                            st.write(f"Avg Rating: {row['avg_rating']:.2f}")
                        
                        with col4:
                            st.write(f"Ratings: {row['rating_count']}")
                else:
                    st.warning("Unable to retrieve popular movie data.")
    
    # Tab 2: User Profile
    with tab2:
        st.header(f"User {user_id} Profile Analysis")
        
        # Get user rating data
        user_ratings = query_user_ratings(user_id)
        
        if user_ratings is not None and not user_ratings.empty:
            # User rating statistics
            rating_counts = user_ratings['rating'].value_counts().sort_index()
            avg_rating = user_ratings['rating'].mean()
            
            # Display basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rated Movies", len(user_ratings))
            with col2:
                st.metric("Average Rating", round(avg_rating, 2))
            with col3:
                st.metric("Most Common Rating", rating_counts.idxmax())
            
            # Display rating distribution
            st.subheader("User Rating Distribution")
            fig = px.histogram(
                user_ratings,
                x='rating',
                nbins=10,
                color_discrete_sequence=['#1f77b4'],
                labels={'rating': 'Rating Value', 'count': 'Count'},
                title=f"User {user_id} Rating Distribution"
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)
            
            # User's highest rated movies with clickable titles
            st.subheader("User's Favorite Movies")
            top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
            
            for _, row in top_rated.iterrows():
                movie_details = query_movie_details(row['movieId'])
                if movie_details is not None and not movie_details.empty:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        # Make title clickable
                        movie_id = row['movieId']
                        if st.button(f"{movie_details.iloc[0]['title']}", key=f"fav_{movie_id}"):
                            st.session_state.show_movie_detail = True
                            st.session_state.selected_movie_id = movie_id
                            st.rerun()
                    
                    with col2:
                        st.write(movie_details.iloc[0]['genres'])
                    
                    with col3:
                        st.write(f"Your Rating: {row['rating']}")
            
            # User genre preference analysis
            st.subheader("User Genre Preferences")
            genre_fig = create_user_genre_heatmap(user_id)
            if genre_fig:
                st.plotly_chart(genre_fig)
            else:
                st.info("Unable to generate genre preference analysis, possibly insufficient data.")
                
            # User tag analysis
            try:
                from db_connect import get_db_connection
                conn = get_db_connection()
                if conn is not None:
                    # Query user tag data
                    tag_query = f'SELECT tag, COUNT(*) as count FROM tags WHERE "userId" = {user_id} GROUP BY tag ORDER BY count DESC LIMIT 10;'
                    user_tags = pd.read_sql_query(tag_query, conn)
                    conn.close()
                    
                    if not user_tags.empty:
                        st.subheader("User Frequently Used Tags")
                        fig = px.bar(
                            user_tags,
                            x='tag',
                            y='count',
                            title="User Frequently Used Tags",
                            labels={'tag': 'Tag', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig)
            except Exception as e:
                # If tag data not available, don't display error
                pass
        else:
            st.warning(f"No rating records found for user {user_id}.")
    
    # Tab 3: Movie Explorer
    with tab3:
        st.header("Movie Library Explorer")
        
        # Load movie data
        movies_df = get_movies()
        
        if not movies_df.empty:
            # Search and filter
            search_query = st.text_input("Search Movie Titles", "")
            
            # Genre filter
            genre_options = get_movie_genres()
            selected_genres = st.multiselect("Select Movie Genres", options=genre_options)
            
            # Year filter (extract year from movie title)
            movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
            year_min, year_max = 1900, 2023
            if not movies_df['year'].dropna().empty:
                year_min = int(movies_df['year'].dropna().astype(int).min())
                year_max = int(movies_df['year'].dropna().astype(int).max())
            
            year_range = st.slider(
                "Select Year Range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max)
            )
            
            # Apply filters
            filtered_movies = movies_df.copy()
            
            # Title search
            if search_query:
                filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_query, case=False)]
            
            # Genre filter
            if selected_genres:
                genre_mask = filtered_movies['genres'].apply(
                    lambda x: any(genre in x.split('|') for genre in selected_genres)
                )
                filtered_movies = filtered_movies[genre_mask]
            
            # Year filter
            year_mask = (filtered_movies['year'].astype(float) >= year_range[0]) & (filtered_movies['year'].astype(float) <= year_range[1])
            filtered_movies = filtered_movies[year_mask]
            
            # Display results
            if not filtered_movies.empty:
                st.write(f"Found {len(filtered_movies)} movies matching criteria")
                
                # Pagination
                items_per_page = 20
                total_pages = (len(filtered_movies) + items_per_page - 1) // items_per_page
                
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=1
                )
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(filtered_movies))
                
                page_movies = filtered_movies.iloc[start_idx:end_idx]
                
                # Display movies with clickable titles
                for index, row in page_movies.iterrows():
                    col1, col2 = st.columns([3, 3])
                    
                    with col1:
                        # Make title clickable
                        movie_id = row['movieId']
                        if st.button(f"{row['title']}", key=f"exp_{movie_id}"):
                            st.session_state.show_movie_detail = True
                            st.session_state.selected_movie_id = movie_id
                            st.rerun()
                    
                    with col2:
                        st.write(row['genres'])
                
                # Movie genre distribution visualization
                st.subheader("Genre Distribution in Filtered Results")
                genres_list = []
                for genre_str in filtered_movies["genres"]:
                    genres_list.extend(genre_str.split("|"))
                
                genre_counts = pd.Series(genres_list).value_counts().head(10)
                fig = px.bar(
                    x=genre_counts.index,
                    y=genre_counts.values,
                    labels={'x': 'Movie Genre', 'y': 'Count'},
                    title="Top 10 Movie Genres"
                )
                st.plotly_chart(fig)
                
                # Year distribution visualization
                st.subheader("Year Distribution in Filtered Results")
                year_counts = filtered_movies['year'].astype(float).value_counts().sort_index()
                fig = px.line(
                    x=year_counts.index,
                    y=year_counts.values,
                    labels={'x': 'Year', 'y': 'Movie Count'},
                    title="Movie Year Distribution Trend"
                )
                st.plotly_chart(fig)
            else:
                st.warning("No movies match the criteria.")
        else:
            st.error("Unable to load movie data.")
    
    # Tab 4: System Performance
    with tab4:
        st.header("Recommendation System Performance")
        
        if model is not None:
            # Display model information
            st.subheader("Model Parameters")
            model_info = {
                "Model Type": "SVD Model" if model.use_svd else "Traditional Collaborative Filtering",
                "Similarity Method": model.sim_method,
                "Number of Similar Users": model.n_sim_user,
                "Number of Recommendations": model.n_rec_movie,
                "Training Split Ratio": model.pivot,
                "Training Set User Count": len(model.trainSet) if model.trainSet else 0
            }
            
            if model.use_svd:
                model_info["SVD Factors"] = model.n_factors
            
            # Create table
            st.table(pd.DataFrame([model_info]).T.reset_index().rename(
                columns={"index": "Parameter", 0: "Value"}
            ))
            
            # Add database status information
            st.subheader("Database Status")
            try:
                from db_connect import get_db_connection
                conn = get_db_connection()
                if conn is not None:
                    # Query database table row counts
                    cursor = conn.cursor()
                    
                    # Get row counts for each table
                    cursor.execute("SELECT COUNT(*) FROM movies;")
                    movies_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM ratings;")
                    ratings_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM tags;")
                    tags_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT \"userId\") FROM ratings;")
                    users_count = cursor.fetchone()[0]
                    
                    conn.close()
                    
                    # Display database statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Number of Movies", f"{movies_count:,}")
                    with col2:
                        st.metric("Number of Users", f"{users_count:,}")
                    with col3:
                        st.metric("Rating Records", f"{ratings_count:,}")
                    with col4:
                        st.metric("Tag Records", f"{tags_count:,}")
            except Exception as e:
                st.error(f"Error retrieving database status: {e}")
            
            # Evaluation button
            if st.button("Evaluate System Performance"):
                with st.spinner("Evaluating system performance, this may take some time..."):
                    metrics = model.evaluate_model()
                
                if metrics:
                    # Display evaluation metrics
                    st.subheader("Recommendation System Evaluation Metrics")
                    
                    # Create metrics chart
                    metrics_to_plot = {k: v for k, v in metrics.items() if isinstance(v, float)}
                    
                    fig = go.Figure()
                    for metric, value in metrics_to_plot.items():
                        fig.add_trace(go.Bar(
                            x=[metric],
                            y=[value],
                            name=metric
                        ))
                    
                    fig.update_layout(
                        title="Recommendation System Performance Metrics",
                        xaxis_title="Metric",
                        yaxis_title="Value",
                        yaxis=dict(range=[0, 1]),
                        height=400
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display metric explanations
                    with st.expander("View Metric Explanations"):
                        st.markdown("""
                        - **Precision@K**: Proportion of recommended items that are relevant.
                        - **Recall@K**: Proportion of relevant items that are recommended.
                        - **F1@K**: Harmonic mean of Precision and Recall, balancing both metrics.
                        - **Coverage**: Proportion of all items that the system is able to recommend.
                        """)
            
            # Explain current model recommendation principles
            with st.expander("Learn About Recommendation Principles"):
                if model.use_svd:
                    st.markdown("""
                    ### SVD Matrix Factorization Recommendation Principles
                    
                    SVD (Singular Value Decomposition) is a dimensionality reduction technique used in collaborative filtering:
                    
                    1. The system decomposes the user-movie rating matrix into low-dimensional latent factor representations
                    2. Each user and movie is described by a set of latent factors
                    3. Ratings are predicted using the dot product of these latent factors
                    4. Movies with highest predicted ratings are recommended
                    
                    SVD models capture latent associations between users and movies and handle data sparsity issues.
                    """)
                else:
                    st.markdown("""
                    ### User-Based Collaborative Filtering Principles
                    
                    Traditional collaborative filtering is based on the assumption that "similar users like similar movies":
                    
                    1. The system identifies users with similar rating behavior to the target user
                    2. Similarity is calculated using cosine similarity, Pearson correlation, or Manhattan distance
                    3. Ratings for unwatched movies are predicted based on ratings from similar users
                    4. Movies with highest predicted ratings are recommended
                    
                    This method is intuitive and effective, but computationally expensive with large user bases.
                    """)
                    
        else:
            st.error("Model not loaded, cannot display system performance.")

if __name__ == "__main__":
    main()