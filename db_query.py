from db_connect import get_db_connection
import pandas as pd
# --------------------- 数据库查询函数 ---------------------
def query_movie_details(movie_id):
    """
    根据电影 ID 从 movies 表查询电影详细信息。
    """
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return None
    try:
        query = f'SELECT * FROM movies WHERE "movieId" = {movie_id};'
        movie_details = pd.read_sql_query(query, conn)
        return movie_details
    except Exception as e:
        print("查询电影详情时出错：", e)
        return None
    finally:
        conn.close()

def query_user_ratings(user_id):
    """
    根据用户 ID 从 ratings 表查询该用户的评分记录。
    """
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return None
    try:
        query = f'SELECT * FROM ratings WHERE "userId" = {user_id};'
        user_ratings = pd.read_sql_query(query, conn)
        return user_ratings
    except Exception as e:
        print("查询用户评分时出错：", e)
        return None
    finally:
        conn.close()

def query_all_movies():
    """
    查询所有电影信息。
    """
    print("test")
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return pd.DataFrame()  
    try:
        query = 'SELECT * FROM movies;'
        movies = pd.read_sql_query(query, conn)
        return movies
    except Exception as e:
        print("查询所有电影时出错：", e)
        return pd.DataFrame()
    finally:
        conn.close()

def query_all_ratings():
    """
    查询所有用户评分数据。
    """
    conn = get_db_connection()
    if conn is None:
        print("无法建立数据库连接。")
        return pd.DataFrame()
    try:
        query = 'SELECT * FROM ratings;'
        ratings = pd.read_sql_query(query, conn)
        return ratings
    except Exception as e:
        print("查询所有评分数据时出错：", e)
        return pd.DataFrame()
    finally:
        conn.close()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to query and display all data from the movies table.
"""

# import pandas as pd
# import psycopg2
# from db_config import DB_CONFIG

# def get_db_connection():
#     """
#     Establish a connection to the PostgreSQL database using configuration parameters.
#     """
#     try:
#         conn = psycopg2.connect(
#             user=DB_CONFIG["DB_USER"],
#             password=DB_CONFIG["DB_PASSWORD"],
#             host=DB_CONFIG["DB_HOST"],
#             port=DB_CONFIG["DB_PORT"],
#             database=DB_CONFIG["DB_NAME"]
#         )
#         return conn
#     except Exception as e:
#         print(f"Error connecting to PostgreSQL: {e}")
#         return None

# def get_movies_table_structure():
#     """
#     Get the structure of the movies table to understand its columns.
#     """
#     conn = get_db_connection()
#     if conn is None:
#         return
    
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT column_name, data_type
#             FROM information_schema.columns
#             WHERE table_name = 'movies'
#             ORDER BY ordinal_position;
#         """)
        
#         columns = cursor.fetchall()
        
#         print("Movies Table Structure:")
#         print("-" * 40)
#         for name, data_type in columns:
#             print(f"{name}: {data_type}")
#         print("-" * 40)
        
#     except Exception as e:
#         print(f"Error getting table structure: {e}")
    
#     finally:
#         conn.close()

# def query_all_movies():
#     """
#     Query all records from the movies table.
#     """
#     conn = get_db_connection()
#     if conn is None:
#         return
    
#     try:
#         # Use pandas to read the SQL query results
#         query = "SELECT * FROM movies ORDER BY \"movieId\" LIMIT 10000;"
#         df = pd.read_sql_query(query, conn)
        
#         # Get the total count
#         cursor = conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM movies;")
#         total_count = cursor.fetchone()[0]
        
#         # Display information
#         print(f"\nTotal Movies: {total_count}")
#         print(f"Displaying first {min(len(df), 10000)} records")
        
#         if len(df) > 0:
#             # Print basic statistics
#             print("\nBasic Statistics:")
#             print(f"- Earliest Movie ID: {df['movieId'].min()}")
#             print(f"- Latest Movie ID: {df['movieId'].max()}")
            
#             # Check if imdb_url column exists
#             if 'imdb_url' in df.columns:
#                 url_count = df['imdb_url'].notna().sum()
#                 print(f"- Movies with IMDb URLs: {url_count} ({url_count/len(df)*100:.2f}%)")
            
#             # Sample of records
#             print("\nSample Records:")
#             print("-" * 80)
            
#             # Only print first 5 rows for clarity
#             sample = df.head(5)
#             for _, row in sample.iterrows():
#                 print(f"Movie ID: {row['movieId']}")
#                 print(f"Title: {row['title']}")
#                 print(f"Genres: {row['genres']}")
#                 if 'imdb_url' in row and pd.notna(row['imdb_url']):
#                     print(f"IMDb URL: {row['imdb_url']}")
#                 print("-" * 80)
            
#             # Ask if user wants to save to CSV
#             save = input("\nDo you want to save all data to a CSV file? (y/n): ")
#             if save.lower() == 'y':
#                 filename = "movies_data.csv"
#                 df.to_csv(filename, index=False)
#                 print(f"Data saved to {filename}")
#         else:
#             print("No movies found in the database.")
        
#     except Exception as e:
#         print(f"Error querying movies: {e}")
    
#     finally:
#         conn.close()

# def main():
#     print("Querying movies table data...")
    
#     # Get table structure
#     get_movies_table_structure()
    
#     # Query all movies
#     query_all_movies()
    
#     print("\nQuery completed!")

# if __name__ == "__main__":
#     main()