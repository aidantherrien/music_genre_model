import os
import psycopg2
import yt_dlp
import re
import random

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

# Number of songs to download per run
NUM_SONGS_TO_DOWNLOAD = 5

# File storage location
SAVE_PATH = "/current_data/unprocessed"

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

# Keywords to filter out compilations
COMPILATION_KEYWORDS = [
    "mix", "playlist", "compilation", "best of", "top", "remix", "medley",
    "mashup", "festival", "charts", "music 2024", "EDM Party", "summer hits",
    "DJ Set", "vs.", "&", ",", "/"
]

SEARCH_QUERIES = [
    "classic rock hits", "classic hip hop", "jazz standards", "country classics",
    "metal anthems", "reggae classics", "folk music", "edm",
    "classic blues songs", "classical symphonies"
]

PLAYLIST_SEARCH_QUERIES = [
    "best new rock music playlist", "hip hop top hits playlist", "classic jazz songs playlist",
    "top metal songs official", "indie folk official playlist", "reggae hits playlist",
    "country music top songs playlist", "electronic music playlist official"
]


def is_valid_song(title):
    """Checks if a video title likely represents a single song and not a compilation."""
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in COMPILATION_KEYWORDS):
        return False
    if "," in title or "&" in title:
        return False
    return True


def extract_genre(video_info):
    """Extracts genre from video metadata."""
    genre_keywords = [
        "rock", "pop", "hip hop", "jazz", "classical", "country", "metal", "reggae",
        "blues", "disco", "electronic", "indie", "folk", "r&b", "funk", "punk",
        "gospel", "house", "techno", "trap", "bluegrass"
    ]
    genre_pattern = re.compile(r"(?i)\b(" + "|".join(genre_keywords) + r")\b")

    for field in ["tags", "description", "title"]:
        if field in video_info and video_info[field]:
            match = genre_pattern.search(str(video_info[field]))
            if match:
                return match.group(1).lower()
    return None


def download_song(video_url, video_id):
    """Downloads the audio from a YouTube video."""
    output_file = os.path.join(SAVE_PATH, f"{video_id}.%(ext)s")
    result = os.system(f'yt-dlp -x --audio-format mp3 -o "{output_file}" "{video_url}"')
    final_output = os.path.join(SAVE_PATH, f"{video_id}.mp3")
    return os.path.exists(final_output)


def fetch_songs_from_search():
    """Searches YouTube for individual songs."""
    query = random.choice(SEARCH_QUERIES)
    ydl_opts = {"quiet": False, "default_search": f"ytsearch10:{query}", "extract_flat": False}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(query, download=False)
        if not results or "entries" not in results:
            return []

        valid_songs = []
        for video in results["entries"]:
            if not video:
                continue

            title, video_id = video.get("title"), video.get("id")
            duration, views = video.get("duration", 0), video.get("view_count", 0)

            if not is_valid_song(title) or duration < 90 or duration > 420 or views < 50000:
                continue

            cursor.execute("SELECT video_id FROM songs WHERE video_id = %s", (video_id,))
            if cursor.fetchone():
                continue

            genre = extract_genre(video)
            if not genre:
                continue

            valid_songs.append(
                {"video_id": video_id, "title": title, "artist": video.get("uploader", "Unknown"), "genre": genre,
                 "url": f"https://www.youtube.com/watch?v={video_id}"})

        return valid_songs


def fetch_songs_from_playlist():
    """Searches YouTube for playlists and extracts individual songs."""
    query = random.choice(PLAYLIST_SEARCH_QUERIES)
    ydl_opts = {"quiet": False, "default_search": f"ytsearch5:{query}", "extract_flat": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(query, download=False)
        if not results or "entries" not in results:
            return []

        playlists = [p for p in results["entries"] if p.get("url")]
        if not playlists:
            return []

        playlist_url = random.choice(playlists)["url"]
        ydl_opts = {"quiet": False, "extract_flat": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_results = ydl.extract_info(playlist_url, download=False)
            if not playlist_results or "entries" not in playlist_results:
                return []

            return [
                {"video_id": v.get("id"), "title": v.get("title"), "artist": v.get("uploader", "Unknown"),
                 "genre": extract_genre(v), "url": f"https://www.youtube.com/watch?v={v.get('id')}"}
                for v in playlist_results["entries"] if v and is_valid_song(v.get("title"))
            ]


# Main execution
found_songs = []
while len(found_songs) < NUM_SONGS_TO_DOWNLOAD:
    found_songs.extend(fetch_songs_from_search())
    if len(found_songs) < NUM_SONGS_TO_DOWNLOAD:
        found_songs.extend(fetch_songs_from_playlist())

found_songs = found_songs[:NUM_SONGS_TO_DOWNLOAD]

for song in found_songs:
    if download_song(song["url"], song["video_id"]):
        cursor.execute(
            "INSERT INTO songs (video_id, title, artist, genre, downloaded) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (video_id) DO NOTHING",
            (song["video_id"], song["title"], song["artist"], song["genre"], True))
        conn.commit()
        print(f"Downloaded and saved: {song['title']}")