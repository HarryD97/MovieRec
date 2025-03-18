#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to add IMDb URLs to movies table by searching for movie titles.
Uses both the links table when available and OMDB API for titles without links.
"""

import pandas as pd
import psycopg2
import requests
import time
import re
from db_config import DB_CONFIG

# You'll need to get a free API key from http://www.omdbapi.com/
# Free tier allows 1,000 requests per day
OMDB_API_KEY = "YOUR_OMDB_API_KEY"  # Replace with your API key

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database using configuration parameters.
    """
    try:
        conn = psycopg2.connect(
            user=DB_CONFIG["DB_USER"],
            password=DB_CONFIG["DB_PASSWORD"],
            host=DB_CONFIG["DB_HOST"],
            port=DB_CONFIG["DB_PORT"],
            database=DB_CONFIG["DB_NAME"]
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def add_imdb_url_column():
    """
    Add the imdb_url column to the movies table if it doesn't already exist.
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if imdb_url column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'movies' AND column_name = 'imdb_url'
        """)
        
        if cursor.fetchone() is None:
            print("Adding imdb_url column to movies table...")
            cursor.execute("""
                ALTER TABLE movies
                ADD COLUMN imdb_url TEXT
            """)
            conn.commit()
            print("Column added successfully.")
        else:
            print("imdb_url column already exists.")
        
        return True
    
    except Exception as e:
        conn.rollback()
        print(f"Error adding column: {e}")
        return False
    
    finally:
        conn.close()

def extract_year_from_title(title):
    """
    Extract the year from a movie title that follows the format "Movie Title (YYYY)"
    """
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return match.group(1)
    return None

def clean_title_for_search(title):
    """
    Clean the movie title for search by removing the year and special characters
    """
    # Remove year in parentheses from title
    clean_title = re.sub(r'\s*\(\d{4}\)$', '', title)
    # Remove special characters that might affect search
    clean_title = re.sub(r'[^\w\s]', '', clean_title)
    return clean_title.strip()

def get_imdb_url_from_api(title):
    """
    Use OMDB API to get IMDb ID for a movie title
    """
    # Clean the title and extract year if available
    year = extract_year_from_title(title)
    clean_title = clean_title_for_search(title)
    
    # Prepare the API request
    params = {
        'apikey': OMDB_API_KEY,
        't': clean_title,  # Search by title
        'type': 'movie',   # Only search for movies
    }
    
    # Add year parameter if available
    if year:
        params['y'] = year
    
    try:
        response = requests.get('http://www.omdbapi.com/', params=params)
        data = response.json()
        
        # Check if the request was successful and an IMDb ID was found
        if data.get('Response') == 'True' and 'imdbID' in data:
            imdb_id = data['imdbID']
            imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
            return imdb_url
        else:
            # If exact title match failed, try search
            params = {
                'apikey': OMDB_API_KEY,
                's': clean_title,  # Search by title
                'type': 'movie',   # Only search for movies
            }
            if year:
                params['y'] = year
                
            response = requests.get('http://www.omdbapi.com/', params=params)
            data = response.json()
            
            if data.get('Response') == 'True' and 'Search' in data and len(data['Search']) > 0:
                imdb_id = data['Search'][0]['imdbID']
                imdb_url = f"https://www.imdb.com/title/{imdb_id}/"
                return imdb_url
            
        return None
    except Exception as e:
        print(f"Error searching for '{title}': {e}")
        return None

def update_movies_with_imdb_urls():
    """
    Update the movies table with IMDb URLs using both links table and API search
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Step 1: Update movies that have IMDb IDs in the links table
        print("Updating movies with IMDb IDs from links table...")
        cursor.execute("""
            UPDATE movies m
            SET imdb_url = 'https://www.imdb.com/title/tt' || LPAD(l."imdbId"::text, 7, '0') || '/'
            FROM links l
            WHERE m."movieId" = l."movieId"
              AND m.imdb_url IS NULL
              AND l."imdbId" IS NOT NULL;
        """)
        
        updated_from_links = cursor.rowcount
        conn.commit()
        print(f"Updated {updated_from_links} movies using links table.")
        
        # Step 2: Find movies that still don't have IMDb URLs
        cursor.execute("""
            SELECT "movieId", title
            FROM movies
            WHERE imdb_url IS NULL
            ORDER BY "movieId"
        """)
        
        movies_to_search = cursor.fetchall()
        print(f"Found {len(movies_to_search)} movies without IMDb URLs, will search API.")
        
        if not movies_to_search:
            print("No movies need to be searched via API.")
            return True
        
        # Step 3: Use API to search for remaining movies
        api_updates = 0
        api_failures = 0
        
        for movie_id, title in movies_to_search:
            print(f"Searching for: {title}")
            imdb_url = get_imdb_url_from_api(title)
            
            if imdb_url:
                cursor.execute(
                    'UPDATE movies SET imdb_url = %s WHERE "movieId" = %s',
                    (imdb_url, movie_id)
                )
                api_updates += 1
                print(f"Found IMDb URL: {imdb_url}")
            else:
                api_failures += 1
                print(f"Failed to find IMDb URL for: {title}")
            
            # Commit every 10 updates to avoid losing all progress if something fails
            if api_updates % 10 == 0:
                conn.commit()
                print(f"Progress: {api_updates} updated, {api_failures} failed")
            
            # Sleep briefly to avoid API rate limits
            time.sleep(0.5)
        
        # Final commit
        conn.commit()
        
        print(f"\nSummary:")
        print(f"- Updated {updated_from_links} movies using links table")
        print(f"- Updated {api_updates} movies using API search")
        print(f"- Failed to find {api_failures} movies")
        
        return True
    
    except Exception as e:
        conn.rollback()
        print(f"Error updating IMDb URLs: {e}")
        return False
    
    finally:
        conn.close()

def verify_imdb_urls():
    """
    Verify a sample of movies to ensure IMDb URLs were added correctly.
    """
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        
        # Sample from links table
        cursor.execute("""
            SELECT m."movieId", m.title, m.imdb_url
            FROM movies m
            JOIN links l ON m."movieId" = l."movieId"
            WHERE m.imdb_url IS NOT NULL
            LIMIT 3
        """)
        
        link_sample = cursor.fetchall()
        
        # Sample from API search (those without links but with URLs)
        cursor.execute("""
            SELECT m."movieId", m.title, m.imdb_url
            FROM movies m
            LEFT JOIN links l ON m."movieId" = l."movieId"
            WHERE m.imdb_url IS NOT NULL
            AND l."movieId" IS NULL
            LIMIT 3
        """)
        
        api_sample = cursor.fetchall()
        
        if link_sample:
            print("\nSample of movies with IMDb URLs from links table:")
            for movie_id, title, imdb_url in link_sample:
                print(f"Movie ID: {movie_id}")
                print(f"Title: {title}")
                print(f"IMDb URL: {imdb_url}")
                print("-" * 50)
        
        if api_sample:
            print("\nSample of movies with IMDb URLs from API search:")
            for movie_id, title, imdb_url in api_sample:
                print(f"Movie ID: {movie_id}")
                print(f"Title: {title}")
                print(f"IMDb URL: {imdb_url}")
                print("-" * 50)
        
        if not link_sample and not api_sample:
            print("No movies with IMDb URLs found.")
        
    except Exception as e:
        print(f"Error verifying IMDb URLs: {e}")
    
    finally:
        conn.close()

def main():
    print("Starting to add IMDb URLs to movies...")
    
    # Step 1: Add imdb_url column
    if not add_imdb_url_column():
        print("Failed to add imdb_url column. Exiting.")
        return
    
    # Step 2: Update movies with IMDb URLs (using both links table and API)
    if not update_movies_with_imdb_urls():
        print("Failed to update movies with IMDb URLs. Exiting.")
        return
    
    # Step 3: Verify a sample of movies
    verify_imdb_urls()
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()