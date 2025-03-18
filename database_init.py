import psycopg2
from psycopg2 import sql, Error
import os
import pandas as pd
from sqlalchemy import create_engine, text
from db_config import DB_CONFIG

data_path = '/home/harry/eep/EEP567/CourseProject/ml-32m'

# 2. connect to PostgreSQL database
def create_engine_postgres():
    DB_USER     = DB_CONFIG["DB_USER"]
    DB_PASSWORD = DB_CONFIG["DB_PASSWORD"]
    DB_HOST     = DB_CONFIG["DB_HOST"]
    DB_PORT     = DB_CONFIG["DB_PORT"]
    DB_NAME     = DB_CONFIG["DB_NAME"]
    
    engine_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_str)
    return engine

# 3. create tables
def create_tables(engine):
    create_movies_table = """
    CREATE TABLE IF NOT EXISTS movies (
        movieId INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        genres TEXT NOT NULL
    );
    """
    create_ratings_table = """
    CREATE TABLE IF NOT EXISTS ratings (
        userId INTEGER NOT NULL,
        movieId INTEGER NOT NULL,
        rating REAL NOT NULL,
        timestamp INTEGER NOT NULL,
        PRIMARY KEY (userId, movieId),
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    );
    """
    create_tags_table = """
    CREATE TABLE IF NOT EXISTS tags (
        userId INTEGER NOT NULL,
        movieId INTEGER NOT NULL,
        tag TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        PRIMARY KEY (userId, movieId, tag),
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    );
    """
    create_links_table = """
    CREATE TABLE IF NOT EXISTS links (
        movieId INTEGER PRIMARY KEY,
        imdbId INTEGER,
        tmdbId INTEGER,
        FOREIGN KEY (movieId) REFERENCES movies(movieId)
    );
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_movies_table))
            conn.execute(text(create_ratings_table))
            conn.execute(text(create_tags_table))
            conn.execute(text(create_links_table))
            print("Successfully created tables in PostgreSQL.")
        except Exception as e:
            print("Failed to create tables:", e)

# 4. Insert data using chunk reading (each chunk contains 10,000 rows)
def insert_csv_in_chunks(engine, table_name, file_path, chunksize=10000):
    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
            # 使用 method='multi' 可以加速批量插入
            chunk.to_sql(table_name, engine, if_exists='append', index=False, method='multi')
            print(f"Chunk {i+1} inserted into {table_name} table.")
    except Exception as e:
        print(f"Failed to insert data into {table_name}: {e}")

# 5. main function
def main():
    engine = create_engine_postgres()
    print("Creating tables...")
    create_tables(engine)
    
    # 定义各文件路径
    movies_fp  = os.path.join(data_path, 'movies.csv')
    ratings_fp = os.path.join(data_path, 'ratings.csv')
    tags_fp    = os.path.join(data_path, 'tags.csv')
    links_fp   = os.path.join(data_path, 'links.csv')
    
    print("Inserting data into movies table...")
    insert_csv_in_chunks(engine, "movies", movies_fp)
    
    print("Inserting data into ratings table...")
    insert_csv_in_chunks(engine, "ratings", ratings_fp)
    
    print("Inserting data into tags table...")
    insert_csv_in_chunks(engine, "tags", tags_fp)
    
    print("Inserting data into links table...")
    insert_csv_in_chunks(engine, "links", links_fp)
    
    print("Data insertion completed.")

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

if __name__ == "__main__":
    main()
    initialize_database_indexes()