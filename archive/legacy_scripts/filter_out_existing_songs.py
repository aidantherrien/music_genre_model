import os
import csv
import psycopg2

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

# Paths for the CSV
CSV_PATH = "/notebooks"
SONG_LIST_CSV = os.path.join(CSV_PATH, "historical_songs.csv")


def get_songs_in_db():
    """Fetches all song titles from the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT song_title FROM songs")  # Assuming table is 'songs' and column is 'song_title'
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    # Return a list of song titles in the database
    return {song[0] for song in result}  # Using a set for faster lookup


def load_songs_from_csv():
    """Loads songs from the CSV into a list."""
    songs = []
    with open(SONG_LIST_CSV, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        songs = [row[0] for row in reader]
    return songs


def save_filtered_songs_to_csv(filtered_songs):
    """Saves the filtered song list to a new CSV file."""
    filtered_csv_path = os.path.join(CSV_PATH, "filtered_historical_songs.csv")
    with open(filtered_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Song Title"])  # Header row
        for song in filtered_songs:
            writer.writerow([song])  # Each song in a new row


def filter_songs():
    """Filters out songs that are already in the database from the CSV."""
    # Load songs from CSV
    songs_from_csv = load_songs_from_csv()

    # Get song titles from the database
    songs_in_db = get_songs_in_db()

    # Filter out songs that are already in the database
    filtered_songs = [song for song in songs_from_csv if song not in songs_in_db]

    # Save the filtered songs to a new CSV
    save_filtered_songs_to_csv(filtered_songs)
    print(f"Saved {len(filtered_songs)} unique songs to filtered_historical_songs.csv.")


# Run the filtering function
filter_songs()
