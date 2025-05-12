import psycopg2

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",   # Database name
    "user": "genre_user",        # User (or the new user you created, 'genre_user')
    "password": "joebiden", # Your PostgreSQL password, joebiden
    "host": "localhost",       # Host (default is localhost)
    "port": 5432               # Port (default is 5432)
}

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("Connected to PostgreSQL!")

    # Example query: List all tables in the public schema
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cursor.fetchall()
    print("Tables:", tables)

    cursor.close()
    conn.close()
    print("Connection closed.")
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
