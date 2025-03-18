import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sample_ratings():
    """Get sample ratings data for analysis"""
    try:
        # Use SQL to directly limit returned results, avoiding loading the entire ratings table
        conn = get_db_connection()
        if conn is None:
            return pd.DataFrame()
        query = 'SELECT * FROM ratings ORDER BY random() LIMIT 100000;'
        ratings_sample = pd.read_sql_query(query, conn)
        conn.close()
        return ratings_sample
    except Exception as e:
        st.error(f"Error getting ratings sample: {e}")
        return pd.DataFrame()

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

# Main application
def main():
    # Initialize session state for ratings if it doesn't exist
    if 'rating_status' not in st.session_state:
        st.session_state.rating_status = None
    
    if 'rating_message' not in st.session_state:
        st.session_state.rating_message = ""
    
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
                st.rerun()  # Replace st.experimental_rerun()
        
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
                        
                        # Display results with rating functionality
                        if not filtered_df.empty:
                            # Create two columns for ratings
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Display recommendation table
                                st.dataframe(
                                    filtered_df[["Movie ID", "Movie Title", "Genres", "Score", "Current Rating"]],
                                    column_config={
                                        "Movie Title": st.column_config.TextColumn("Movie Title", width="large"),
                                        "Genres": st.column_config.TextColumn("Genres", width="medium"),
                                        "Score": st.column_config.ProgressColumn(
                                            "Recommendation Score",
                                            format="%.2f",
                                            min_value=0,
                                            max_value=filtered_df["Score"].max() if not filtered_df.empty else 1
                                        ),
                                        "Current Rating": st.column_config.NumberColumn(
                                            "Your Rating",
                                            format="%.1f ★"
                                        )
                                    },
                                    hide_index=True
                                )
                            
                            # Rating section
                            st.subheader("Rate Movies")
                            st.markdown("Select a movie and provide your rating:")
                            
                            # Movie selection for rating
                            movie_options = {f"{row['Movie ID']} - {row['Movie Title']}": row['Movie ID'] 
                                           for _, row in filtered_df.iterrows()}
                            
                            selected_movie_key = st.selectbox(
                                "Select Movie to Rate",
                                options=list(movie_options.keys())
                            )
                            
                            selected_movie_id = movie_options[selected_movie_key]
                            current_rating = next((row["Current Rating"] 
                                                  for _, row in filtered_df.iterrows() 
                                                  if row["Movie ID"] == selected_movie_id), None)
                            
                            # Rating widget
                            user_rating = create_rating_widget(selected_movie_id, current_rating)
                            
                            # Submit rating button
                            if st.button("Submit Rating"):
                                try:
                                    user_id_int = int(float(user_id))
                                    movie_id_int = int(float(selected_movie_id))
                                    
                                    # Add or update rating in database
                                    success, message = add_or_update_rating(
                                        user_id_int, 
                                        movie_id_int, 
                                        user_rating
                                    )
                                    
                                    # Store status in session state
                                    st.session_state.rating_status = success
                                    st.session_state.rating_message = message
                                    
                                    # Force refresh
                                    st.rerun()  # Replace st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error submitting rating: {str(e)}")
                            
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
                
                # Display popular movies with rating functionality
                popular_movies = get_popular_movies(min_ratings=50, top_n=10)
                if not popular_movies.empty:
                    # Display popular movies
                    st.dataframe(
                        popular_movies,
                        column_config={
                            "movieId": st.column_config.NumberColumn("Movie ID"),
                            "title": st.column_config.TextColumn("Movie Title", width="large"),
                            "genres": st.column_config.TextColumn("Genres", width="medium"),
                            "avg_rating": st.column_config.NumberColumn("Average Rating", format="%.2f"),
                            "rating_count": st.column_config.NumberColumn("Number of Ratings")
                        },
                        hide_index=True
                    )
                    
                    # Rating section for popular movies
                    st.subheader("Rate Popular Movies")
                    st.markdown("Rate some popular movies to get personalized recommendations:")
                    
                    # Movie selection for rating popular movies
                    popular_movie_options = {f"{row['movieId']} - {row['title']}": row['movieId'] 
                                           for _, row in popular_movies.iterrows()}
                    
                    selected_popular_movie_key = st.selectbox(
                        "Select Movie to Rate",
                        options=list(popular_movie_options.keys())
                    )
                    
                    selected_popular_movie_id = popular_movie_options[selected_popular_movie_key]
                    current_pop_rating = user_ratings.get(str(selected_popular_movie_id), None)
                    
                    # Rating widget for popular movies
                    popular_user_rating = create_rating_widget(selected_popular_movie_id, current_pop_rating)
                    
                    # Submit rating button for popular movies
                    if st.button("Submit Rating"):
                        try:
                            user_id_int = int(float(user_id))
                            movie_id_int = int(float(selected_popular_movie_id))
                            
                            # Add or update rating in database
                            success, message = add_or_update_rating(
                                user_id_int, 
                                movie_id_int, 
                                popular_user_rating
                            )
                            
                            # Store status in session state
                            st.session_state.rating_status = success
                            st.session_state.rating_message = message
                            
                            # Force refresh
                            st.rerun()  # Replace st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error submitting rating: {str(e)}")
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
            
            # User's highest rated movies
            st.subheader("User's Favorite Movies")
            top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
            top_movies_data = []
            
            for _, row in top_rated.iterrows():
                movie_details = query_movie_details(row['movieId'])
                if movie_details is not None and not movie_details.empty:
                    top_movies_data.append({
                        "Movie ID": row['movieId'],
                        "Movie Title": movie_details.iloc[0]['title'],
                        "Genres": movie_details.iloc[0]['genres'],
                        "User Rating": row['rating']
                    })
            
            if top_movies_data:
                st.dataframe(
                    pd.DataFrame(top_movies_data),
                    column_config={
                        "Movie Title": st.column_config.TextColumn("Movie Title", width="large"),
                        "Genres": st.column_config.TextColumn("Genres", width="medium"),
                        "User Rating": st.column_config.ProgressColumn(
                            "Rating",
                            format="%.1f",
                            min_value=0,
                            max_value=5
                        )
                    },
                    hide_index=True
                )
                
                # Add ability to update ratings
                st.subheader("Update Your Ratings")
                
                # Movie selection for updating ratings
                rated_movie_options = {f"{row['Movie ID']} - {row['Movie Title']}": row['Movie ID'] 
                                     for _, row in pd.DataFrame(top_movies_data).iterrows()}
                
                selected_rated_movie_key = st.selectbox(
                    "Select Movie to Update Rating",
                    options=list(rated_movie_options.keys())
                )
                
                selected_rated_movie_id = rated_movie_options[selected_rated_movie_key]
                current_rated_rating = next((row["User Rating"] 
                                           for _, row in pd.DataFrame(top_movies_data).iterrows() 
                                           if row["Movie ID"] == selected_rated_movie_id), None)
                
                # Rating widget for updating
                updated_rating = create_rating_widget(selected_rated_movie_id, current_rated_rating)
                
                # Submit updated rating button
                if st.button("Update Rating"):
                    try:
                        user_id_int = int(float(user_id))
                        movie_id_int = int(float(selected_rated_movie_id))
                        
                        # Update rating in database
                        success, message = add_or_update_rating(
                            user_id_int, 
                            movie_id_int, 
                            updated_rating
                        )
                        
                        # Store status in session state
                        st.session_state.rating_status = success
                        st.session_state.rating_message = message
                        
                        # Force refresh
                        st.rerun()  # Replace st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error updating rating: {str(e)}")
            
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
                items_per_page = 50
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
                
                # Display movies table
                st.dataframe(
                    page_movies[['movieId', 'title', 'genres']],
                    column_config={
                        "movieId": st.column_config.NumberColumn("Movie ID"),
                        "title": st.column_config.TextColumn("Movie Title", width="large"),
                        "genres": st.column_config.TextColumn("Genres", width="medium")
                    }
                )
                
                # Add rating functionality for explorer tab
                st.subheader("Rate Movies from Explorer")
                
                # Get user's existing ratings
                user_ratings_df = query_user_ratings(user_id)
                user_ratings = {}
                if user_ratings_df is not None and not user_ratings_df.empty:
                    for _, row in user_ratings_df.iterrows():
                        user_ratings[str(row['movieId'])] = row['rating']
                
                # Movie selection for rating from explorer
                explorer_movie_options = {f"{row['movieId']} - {row['title']}": row['movieId'] 
                                        for _, row in page_movies.iterrows()}
                
                selected_explorer_movie_key = st.selectbox(
                    "Select Movie to Rate",
                    options=list(explorer_movie_options.keys())
                )
                
                selected_explorer_movie_id = explorer_movie_options[selected_explorer_movie_key]
                current_explorer_rating = user_ratings.get(str(selected_explorer_movie_id), None)
                
                # Rating widget for explorer
                explorer_user_rating = create_rating_widget(selected_explorer_movie_id, current_explorer_rating)
                
                # Submit rating button for explorer
                if st.button("Submit Explorer Rating"):
                    try:
                        user_id_int = int(float(user_id))
                        movie_id_int = int(float(selected_explorer_movie_id))
                        
                        # Add or update rating in database
                        success, message = add_or_update_rating(
                            user_id_int, 
                            movie_id_int, 
                            explorer_user_rating
                        )
                        
                        # Store status in session state
                        st.session_state.rating_status = success
                        st.session_state.rating_message = message
                        
                        # Force refresh
                        st.rerun()  # Replace st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error submitting rating: {str(e)}")
                
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