import os
import csv
import random
import psycopg2
import yt_dlp
import requests
from bs4 import BeautifulSoup

# Database configuration
DB_CONFIG = {
    "dbname": "genre_model",
    "user": "genre_user",
    "password": "joebiden",
    "host": "localhost",
    "port": 5432
}

SAVE_PATH = "/currentData/unprocessed"
CSV_PATH = "/notebooks"
SONG_LIST_CSV = os.path.join(CSV_PATH, "historical_songs.csv")

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(CSV_PATH, exist_ok=True)

# Wikipedia and Rolling Stone URLs (Example)
WIKI_URL = "https://en.wikipedia.org/wiki/Rolling_Stone's_500_Greatest_Songs_of_All_Time"


def scrape_wikipedia_song_list(url):
    """Scrapes a Wikipedia page for song titles and artists."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    songs = []

    for row in soup.select("table.wikitable tbody tr")[1:]:
        columns = row.find_all("td")
        if len(columns) >= 2:
            title = columns[1].get_text(strip=True).strip('"')
            artist = columns[2].get_text(strip=True)
            songs.append(f"{title} - {artist}")

    return songs


def save_song_list_to_csv(song_list):
    """Saves the song list to a CSV file."""
    with open(SONG_LIST_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Song Title"])
        for song in song_list:
            writer.writerow([song])


def load_songs_from_csv():
    """Loads songs from CSV into a Python list."""
    songs = []
    with open(SONG_LIST_CSV, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        songs = [row[0] for row in reader]
    return songs


def search_old_song():
    """Selects a historical song and searches it on YouTube."""
    songs = load_songs_from_csv()
    song_title = random.choice(songs)
    print(f"Searching for: {song_title}")

    ydl_opts = {
        "quiet": False,
        "default_search": f"ytsearch10:{song_title} official audio",
        "extract_flat": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(song_title, download=False)
        if not results or "entries" not in results:
            print("No search results found!")
            return None

        for video in results["entries"]:
            if video:
                return {
                    "video_id": video.get("id"),
                    "title": video.get("title"),
                    "url": f"https://www.youtube.com/watch?v={video.get('id')}"
                }
    return None


def download_song(video_url, video_id):
    """Downloads the song from YouTube."""
    output_file = os.path.join(SAVE_PATH, f"{video_id}.%(ext)s")
    result = os.system(f'yt-dlp -x --audio-format mp3 -o "{output_file}" "{video_url}"')
    final_output = os.path.join(SAVE_PATH, f"{video_id}.mp3")
    return os.path.exists(final_output)


# Scrape song list and save to CSV
historical_songs = scrape_wikipedia_song_list(WIKI_URL)

print(historical_songs)

save_song_list_to_csv(historical_songs)

# Search and download song
song = search_old_song()
if song:
    success = download_song(song["url"], song["video_id"])
    if success:
        print(f"Downloaded {song['title']} successfully!")
    else:
        print(f"Failed to download {song['title']}")
else:
    print("No suitable song found.")
