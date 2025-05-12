import csv
import psycopg2
import musicbrainzngs
from datetime import datetime

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

# Paths
CSV_PATH = "/notebooks/filtered_historical_songs.csv"

# Configure MusicBrainz API
musicbrainzngs.set_useragent("GenreModel", "1.0", "aidanmtherrien@gmail.com")


def get_song_metadata(song_title, artist_name):
    """Fetches song metadata (title, artist, genre, year) from MusicBrainz."""
    try:
        result = musicbrainzngs.search_recordings(song_title, artist=artist_name, limit=1)

        if result and "recording-list" in result and result["recording-list"]:
            recording = result["recording-list"][0]

            # Extract song title and artist
            title = recording.get("title", song_title)
            artist = artist_name

            # Extract year from the first release date
            year = "Unknown"
            if "release-list" in recording and recording["release-list"]:
                release_date = recording["release-list"][0].get("date", "")
                if release_date:
                    year = release_date[:4]  # Extract only the year

            # Fetch genre from the artist (MusicBrainz does not store it in recording metadata)
            genre = "Unknown"
            if "artist-credit" in recording and recording["artist-credit"]:
                artist_id = recording["artist-credit"][0].get("artist", {}).get("id")
                if artist_id:
                    artist_data = musicbrainzngs.get_artist_by_id(artist_id, includes=["tags"])
                    tags = artist_data.get("artist", {}).get("tag-list", [])
                    if tags:
                        genre = tags[0]["name"]  # Use first tag as genre

            return {"song_title": title, "artist": artist, "genre": genre, "year": year}

    except Exception as e:
        print(f"Error fetching MusicBrainz data for {song_title} by {artist_name}: {e}")

    return {"song_title": song_title, "artist": artist_name, "genre": "Unknown", "year": "Unknown"}


def song_exists(cursor, song_title, artist):
    """Checks if the song already exists in the database."""
    cursor.execute("""
        SELECT 1 FROM songs WHERE song_title = %s AND artist = %s
    """, (song_title, artist))
    return cursor.fetchone() is not None


def save_metadata_to_db(metadata):
    """Saves metadata to PostgreSQL if the song doesn't already exist."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Avoid inserting duplicates
        if song_exists(cursor, metadata["song_title"], metadata["artist"]):
            print(f"Skipping {metadata['song_title']} - {metadata['artist']} (Already in DB)")
        else:
            cursor.execute("""
                INSERT INTO songs (video_title, song_title, artist, genre, year, downloaded, added_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                metadata["song_title"],  # Use song title as video title
                metadata["song_title"],
                metadata["artist"],
                metadata["genre"],
                metadata["year"],
                False,  # Set downloaded to False
                datetime.now()
            ))
            print(f"Inserted: {metadata['song_title']} - {metadata['artist']}")

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error saving to database: {e}")


def process_songs():
    """Reads the filtered CSV and updates the database with song metadata."""
    with open(CSV_PATH, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            song_info = row[0].split(" - ")
            if len(song_info) == 2:
                song_title, artist_name = song_info
                print(f"Processing: {song_title} by {artist_name}")

                metadata = get_song_metadata(song_title, artist_name)
                save_metadata_to_db(metadata)


if __name__ == "__main__":
    process_songs()
