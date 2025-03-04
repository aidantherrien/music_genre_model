import psycopg2
import yt_dlp
import os
import re

# PostgreSQL connection settings
DB_CONFIG = {
    "dbname": "musicdb",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": "5432"
}

# Base folder where songs are stored
BASE_DOWNLOAD_PATH = r"/currentData/unprocessed"

# Genres to download (excluding "rock")
GENRES = ["jazz", "hip-hop", "classical", "blues", "metal", "pop", "country"]

# Ensure genre subfolders exist
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
    clean_title = sanitize_filename(title)

    genre_folder = os.path.join(BASE_DOWNLOAD_PATH, genre)
    output_template = os.path.join(genre_folder, f"{clean_title}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
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
        if genre in GENRES:  # "rock" is excluded here
            download_song_from_youtube(title, artist, genre)
        else:
            print(f"Skipping {title} - Genre is '{genre}', which is not being downloaded.")
