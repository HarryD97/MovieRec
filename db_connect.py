import psycopg2
from psycopg2 import sql, Error
import pandas as pd
from db_config import DB_CONFIG

def get_db_connection():
    DB_USER     = DB_CONFIG["DB_USER"]
    DB_PASSWORD = DB_CONFIG["DB_PASSWORD"]
    DB_HOST     = DB_CONFIG["DB_HOST"]
    DB_PORT     = DB_CONFIG["DB_PORT"]
    DB_NAME     = DB_CONFIG["DB_NAME"]
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Error as e:
        print(f"Failed at connecting DB: {e}")
        return None
    
def initialize_database_indexes():
    """
    Create database indexes to improve query performance
    This function uses the st.cache_resource decorator to ensure it only executes once when the application first loads
    """
    # Get database connection
    conn = get_db_connection()
    # Create cursor
    cursor = conn.cursor()
    
    # Define indexes to be created
    index_queries = [
        'CREATE INDEX IF NOT EXISTS idx_ratings_userid ON ratings("userId");',
        'CREATE INDEX IF NOT EXISTS idx_ratings_movieid ON ratings("movieId");',
        'CREATE INDEX IF NOT EXISTS idx_tags_userid ON tags("userId");',
        'CREATE INDEX IF NOT EXISTS idx_tags_movieid ON tags("movieId");'
    ]
    
    # Execute index creation
    for query in index_queries:
        cursor.execute(query)
        
    # Commit transaction
    conn.commit()
    
    # Close connection
    cursor.close()
    conn.close()
    print("Successfully create index")
    return True
    
if __name__ == "__main__":
    conn = get_db_connection()
    print(conn)
    if conn is not None:
        print("Successfully connected to PostgreSQL database on Amazon RDS.")
        # initialize_database_indexes()