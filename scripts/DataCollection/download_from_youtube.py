import psycopg2
import yt_dlp
import os
import re

# PostgreSQL connection settings
DB_CONFIG = {
    "dbname": "your_db_name",
    "user": "the_user_you_created",
    "password": "the_password_you_set",
    "host": "localhost",
    "port": "5432"
}

# Base folder where songs will be stored
BASE_DOWNLOAD_PATH = r"path_to_your_currentData_unprocessed_folder"

# Ensure genre subfolders exist
GENRES = ["rock", "jazz", "hip-hop", "classical", "blues", "metal", "pop", "country"]
for genre in GENRES:
    os.makedirs(os.path.join(BASE_DOWNLOAD_PATH, genre), exist_ok=True)


def sanitize_filename(filename):
    """Remove only the characters that are not allowed in filenames."""
    return re.sub(r'[\\/*?:"<>|]', '', filename).strip()


def fetch_songs_from_db():
    """Fetch all song titles, artists, and genres from the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT title, artist, genre FROM songs;")
    songs = cursor.fetchall()
    conn.close()
    return songs


def download_song_from_youtube(title, artist, genre):
    """Search YouTube for a song and download the best match as an MP3 in the appropriate genre folder."""
    query = f"{title} {artist} official audio"
    clean_title = sanitize_filename(title)  # Only removes problematic characters

    genre_folder = os.path.join(BASE_DOWNLOAD_PATH, genre)
    os.makedirs(genre_folder, exist_ok=True)  # Ensure genre folder exists

    output_template = os.path.join(genre_folder, f"{clean_title}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,  # Filename is just the title
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "noplaylist": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.extract_info(f"ytsearch1:{query}", download=True)
            print(f"Downloaded: {clean_title} to {genre_folder}")
        except Exception as e:
            print(f"Failed to download {title}: {e}")


if __name__ == "__main__":
    songs = fetch_songs_from_db()

    for title, artist, genre in songs:
        if genre in GENRES:  # Only process songs with valid genres
            download_song_from_youtube(title, artist, genre)
        else:
            print(f"Skipping {title} - {artist} (Invalid genre: {genre})")
