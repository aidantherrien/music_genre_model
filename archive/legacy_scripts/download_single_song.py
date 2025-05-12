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

# File storage location
SAVE_PATH = "/current_data/unprocessed"
SEARCH_QUERIES = [
    "new single 2024", "official single release", "hit single", "rock official single",
    "pop new single", "hip hop single official video", "electronic new single",
    "indie song official", "top jazz single", "classical piano solo",
    "metal band single", "country music official single", "reggae official single"
]

COMPILATION_KEYWORDS = [
    "mix", "playlist", "compilation", "best of", "top", "remix", "medley",
    "mashup", "festival", "charts", "music 2024", "EDM Party", "summer hits",
    "DJ Set", "vs.", "&", ",", "/"
]


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


def is_valid_song(title):
    """Checks if a video title likely represents a single song and not a compilation."""
    title_lower = title.lower()

    # Filter out compilation keywords
    if any(keyword in title_lower for keyword in COMPILATION_KEYWORDS):
        print(f"Skipping compilation/mix: {title}")
        return False

    # Filter out multiple artist names (e.g., "Lady Gaga, Bruno Mars - XYZ")
    if "," in title or "&" in title:
        print(f"Skipping potential mashup/collab: {title}")
        return False

    return True


# Function to search for a song
def search_song():
    query = random.choice(SEARCH_QUERIES)
    print(f"Searching for: {query}")

    ydl_opts = {
        "quiet": False,
        "default_search": f"ytsearch10:{query}",
        "extract_flat": False,
        "force_generic_extractor": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(query, download=False)
        if not results or "entries" not in results:
            print("No search results found!")
            return None

        random.shuffle(results["entries"])  # Shuffle for variety

        for video in results["entries"]:
            if not video:
                continue

            video_id = video.get("id")
            title = video.get("title")
            artist = video.get("uploader", "Unknown Artist")
            views = video.get("view_count", 0)
            duration = video.get("duration", 0)  # Duration in seconds

            if not is_valid_song(title):
                continue

            if duration < 90 or duration > 420:  # Between 1.5 and 7 minutes
                print(f"Skipping {title} - Duration {duration}s")
                continue

            if views < 50000:
                print(f"Skipping {title} - Only {views} views")
                continue

            genre = extract_genre(video)
            if not genre:
                print(f"Skipping {title} - No genre found")
                continue

            cursor.execute("SELECT video_id FROM songs WHERE video_id = %s", (video_id,))
            if cursor.fetchone():
                print(f"Skipping {title} - Already in database")
                continue

            return {
                "video_id": video_id,
                "title": title,
                "artist": artist,
                "genre": genre,
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }

    return None

# Function to extract genre from metadata
def extract_genre(video_info):
    genre_keywords = [
        "rock", "pop", "hip hop", "jazz", "classical", "country", "metal",
        "reggae", "blues", "disco", "electronic", "indie", "folk", "r&b", "funk",
        "punk", "gospel", "house", "techno", "trap", "bluegrass"
    ]
    genre_pattern = re.compile(r"(?i)\b(" + "|".join(genre_keywords) + r")\b")

    # Check tags
    if "tags" in video_info:
        for tag in video_info["tags"]:
            match = genre_pattern.search(tag)
            if match:
                return match.group(1).lower()

    # Check description
    if "description" in video_info:
        match = genre_pattern.search(video_info["description"])
        if match:
            return match.group(1).lower()

    # Check title
    if "title" in video_info:
        match = genre_pattern.search(video_info["title"])
        if match:
            return match.group(1).lower()

    return None


# Function to download the song
def download_song(video_url, video_id):
    output_file = os.path.join(SAVE_PATH, f"{video_id}.%(ext)s")  # Use template for extension

    # Run yt-dlp to extract audio
    result = os.system(f'yt-dlp -x --audio-format mp3 -o "{output_file}" "{video_url}"')

    # Check if the file exists
    final_output = os.path.join(SAVE_PATH, f"{video_id}.mp3")  # Actual file name

    if os.path.exists(final_output):
        print(f"Download successful: {final_output}")
        return True
    else:
        print(f"Download failed: {final_output} not found.")
        return False

# Main execution
song = search_song()
if song:
    success = download_song(song["url"], song["video_id"])

    if success:
        # Insert into database
        cursor.execute("""
            INSERT INTO songs (video_id, title, artist, genre, downloaded)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO NOTHING
        """, (song["video_id"], song["title"], song["artist"], song["genre"], True))
        conn.commit()

        print(f"Saved {song['title']} to database and downloaded to {SAVE_PATH}")
    else:
        print(f"Failed to download {song['title']}")
else:
    print("No suitable song found.")

# Close database connection
cursor.close()
conn.close()
print("Process complete.")
