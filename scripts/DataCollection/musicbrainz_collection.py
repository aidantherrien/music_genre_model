import psycopg2
import musicbrainzngs
import time
import random

# How many songs per genre you wish to gather
NUM_SONGS = 100

# PostgreSQL connection settings (modify as needed)
DB_CONFIG = {
    "dbname": "your_database_name",
    "user": "the_user_you_made",
    "password": "the_password_you_set",
    "host": "localhost",
    "port": "5432"
}

# Initialize MusicBrainz API
musicbrainzngs.set_useragent("GenreFetcher", "1.0", "youremail@example.com")

def init_db():
    """Initialize the PostgreSQL database and ensure the table exists."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id SERIAL PRIMARY KEY,
            title TEXT,
            artist TEXT,
            genre TEXT,
            trained BOOLEAN DEFAULT FALSE
        );
    """)
    conn.commit()
    conn.close()

def fetch_songs_by_genre(genre, min_songs=100):
    """Fetch at least `min_songs` songs for a given genre from MusicBrainz."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print(f"Fetching songs for genre: {genre}...")

    start = random.randint(0, 500)  # Start at a random place each time for a variety of songs
    step = 25  # MusicBrainz allows max 25 per request
    total_fetched = 0

    while total_fetched < min_songs:
        try:
            result = musicbrainzngs.search_recordings(tag=genre, limit=step, offset=start)
            recordings = result.get("recording-list", [])

            if not recordings:
                print(f"No more songs found for genre: {genre}.")
                break

            for rec in recordings:
                title = rec.get("title", "Unknown Title")
                artist = rec["artist-credit"][0]["name"] if "artist-credit" in rec else "Unknown Artist"

                # Insert into PostgreSQL database
                cursor.execute(
                    "INSERT INTO songs (title, artist, genre) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (title, artist, genre)
                )
                total_fetched += 1

                if total_fetched >= min_songs:
                    break

            conn.commit()
            start += step
            time.sleep(1)  # Respect rate limits

        except Exception as e:
            print(f"Error fetching songs for genre {genre}: {e}")
            break

    conn.close()
    print(f"Fetched {total_fetched} songs for genre: {genre}.")

if __name__ == "__main__":
    init_db()

    # List of genres to fetch
    genres = ["rock", "jazz", "hip-hop", "classical", "blues", "metal", "pop", "country"]

    for genre in genres:
        fetch_songs_by_genre(genre, min_songs=NUM_SONGS)
