import os
import psycopg2
import yt_dlp
import re

# Database Configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

# Path to store downloaded audio files
SAVE_PATH = r"/currentData"
NUM_SONGS = 5  # Number of songs to fetch at a time
SEARCH_QUERY = "official music video"  # Modify to refine search results

# Ensure the save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("Connected to PostgreSQL database!")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Ensure the table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        video_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT,
        genre TEXT,
        downloaded BOOLEAN DEFAULT FALSE,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()


# Function to search and extract video metadata
def search_songs():
    ydl_opts = {
        "quiet": True,
        "default_search": f"ytsearch{NUM_SONGS}",
        "extract_flat": True,
        "force_generic_extractor": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(SEARCH_QUERY, download=False)
        return results.get("entries", []) if results else []


# Function to extract genre from metadata
def extract_genre(video_info):
    genre_pattern = re.compile(r"(rock|pop|hiphop|jazz|classical|country|metal|reggae|blues|disco)", re.I)

    # Check tags
    if "tags" in video_info:
        for tag in video_info["tags"]:
            if genre_pattern.search(tag):
                return tag.lower()

    # Check description
    if "description" in video_info:
        match = genre_pattern.search(video_info["description"])
        if match:
            return match.group(1).lower()

    return None  # No genre found


# Function to download a song
def download_song(video_url, video_id):
    audio_filename = os.path.join(SAVE_PATH, f"{video_id}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_filename,
        "quiet": False,  # Change to False to see download logs
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            # Ensure the file was saved
            if os.path.exists(audio_filename):
                print(f"Download successful: {audio_filename}")
                return True
            else:
                print(f"Download failed: {audio_filename} not found.")
                return False
        except Exception as e:
            print(f"Error downloading {video_url}: {e}")
            return False


# Process songs
songs = search_songs()

for song in songs:
    if not song or "id" not in song or "title" not in song:
        continue

    video_id = song["id"]
    title = song["title"]
    artist = song.get("uploader", "Unknown Artist")
    genre = extract_genre(song)

    # Check if the song is already in the database
    cursor.execute("SELECT downloaded FROM songs WHERE video_id = %s", (video_id,))
    record = cursor.fetchone()

    if record:
        print(f"Skipping {title} - Already processed (Downloaded: {record[0]})")
        continue

    # Skip downloading if no genre is found, but log it
    if not genre:
        cursor.execute("""
            INSERT INTO songs (video_id, title, artist, genre, downloaded)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO NOTHING
        """, (video_id, title, artist, "unknown", False))
        conn.commit()
        print(f"Skipped {title} - No genre found")
        continue

    # Download the song
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    success = download_song(video_url, video_id)

    # Insert/update song in the database
    cursor.execute("""
        INSERT INTO songs (video_id, title, artist, genre, downloaded)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE
        SET downloaded = EXCLUDED.downloaded
    """, (video_id, title, artist, genre, success))
    conn.commit()

    print(f"Processed: {title} - Genre: {genre} - {'Downloaded' if success else 'Failed to download'}")

# Close database connection
cursor.close()
conn.close()
print("Song fetching complete.")
