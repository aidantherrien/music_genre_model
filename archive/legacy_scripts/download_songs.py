import os
import csv
import psycopg2
import yt_dlp
from datetime import datetime
import requests

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

# Paths
SAVE_PATH = "/current_data/unprocessed"
CSV_PATH = "/notebooks"
SONG_LIST_CSV = os.path.join(CSV_PATH, "filtered_historical_songs.csv")

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)


def get_video_metadata(video_url):
    """Extracts metadata from a YouTube video."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": False,
        "force_generic_extractor": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            artist = info.get("uploader", "Unknown")  # Default to 'Unknown' if no artist
            song_title = info.get("title", "Unknown")  # Default to 'Unknown' if no song title

            # Try to get genre from metadata
            genre = info.get("tags")[0] if info.get("tags") else "Unknown"

            # Handle missing genre by putting 'Unknown'
            if genre == "Unknown":
                print(f"Genre not found for {song_title} by {artist}. Using 'Unknown' as genre.")

            metadata = {
                "video_id": info.get("id"),
                "video_title": info.get("title"),
                "song_title": song_title,
                "artist": artist,
                "genre": genre,
                "year": info.get("release_year", "Unknown"),
                "downloaded": False,
                "added_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        return metadata

    except Exception as e:
        print(f"Error fetching metadata for {video_url}: {e}")
        # Return metadata with default values if an error occurs
        return {
            "video_id": "Unknown",
            "video_title": "Unknown",
            "song_title": "Unknown",
            "artist": "Unknown",
            "genre": "Unknown",
            "year": "Unknown",
            "downloaded": False,
            "added_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def download_song(video_url, video_id):
    """Downloads the song from YouTube."""
    output_file = os.path.join(SAVE_PATH, f"{video_id}.%(ext)s")
    result = os.system(f'yt-dlp -x --audio-format mp3 -o "{output_file}" "{video_url}"')
    final_output = os.path.join(SAVE_PATH, f"{video_id}.mp3")
    return os.path.exists(final_output)


def insert_song_metadata(metadata):
    """Inserts song metadata into the SQL database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # SQL insert query
    query = """
    INSERT INTO songs (video_id, video_title, song_title, artist, genre, year, downloaded, added_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (video_id) DO NOTHING;
    """

    # Insert metadata into database
    cursor.execute(query, (
        metadata["video_id"],
        metadata["video_title"],
        metadata["song_title"],
        metadata["artist"],
        metadata["genre"],
        metadata["year"],
        metadata["downloaded"],
        metadata["added_at"]
    ))

    conn.commit()
    cursor.close()
    conn.close()


def search_and_download_song(song_title):
    """Searches YouTube for the song and downloads it."""
    print(f"Searching for: {song_title}")

    ydl_opts = {
        "quiet": True,
        "default_search": f"ytsearch10:{song_title} official audio",
        "extract_flat": False,
        "force_generic_extractor": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(song_title, download=False)
        if not results or "entries" not in results:
            print("No search results found!")
            return None

        for video in results["entries"]:
            if video:
                video_url = f"https://www.youtube.com/watch?v={video.get('id')}"
                return video_url
    return None


def process_songs():
    """Processes all songs from the filtered CSV: search, download, and insert metadata."""
    # Load the filtered song list
    with open(SONG_LIST_CSV, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        songs = [row[0] for row in reader]

    for song_title in songs:
        video_url = search_and_download_song(song_title)
        if video_url:
            video_id = video_url.split("v=")[-1]
            metadata = get_video_metadata(video_url)

            # Download the song
            if download_song(video_url, video_id):
                metadata["downloaded"] = True  # Mark as downloaded
                print(f"Downloaded {metadata['video_title']} successfully!")
            else:
                print(f"Failed to download {metadata['video_title']}")

            # Insert metadata into the database
            insert_song_metadata(metadata)
        else:
            print(f"Skipping {song_title} - No video found.")


# Run the processing function
process_songs()
