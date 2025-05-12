import requests
import json
import csv
import random
import os


# === Last.fm Artist Song Fetcher ===
# This script fetches top tracks for a list of artists grouped by genre using the Last.fm API.
# It then stores metadata about the fetched songs (title, artist, genre) in a CSV file, 
# which will later be used for tasks like audio downloading and dataset creation.
#
# === Dependencies ===
# - requests: for making HTTP requests to the Last.fm API
# - json: for loading the genre-artist mapping and processing API responses
# - csv: for writing the song metadata to a CSV file
# - random: for selecting a random set of top songs for each artist
# - os: for checking file paths and directory existence
#
# === Configuration ===
# API_KEY: Your Last.fm API key (replace with your own)
# ARTIST_JSON_PATH: Path to the JSON file containing artists grouped by genre
# OUTPUT_CSV_PATH: Path to the CSV file where song metadata will be saved
# X_SONGS_PER_ARTIST: Number of songs to fetch per artist
# Y_TOP_SONG_POOL: Number of top tracks to fetch for each artist from Last.fm
#
# === Functions ===
# - fetch_top_tracks: Fetches the top tracks for a given artist from the Last.fm API.
# - main: The main function that loads the genre-artist mapping, fetches songs for each artist,
#         and writes the song metadata to the CSV file.
#
# === Usage ===
# Run this script to fetch the top tracks for each artist, group them by genre, 
# and save the results to a CSV file for further processing.



# === CONFIGURATION ===
API_KEY = r"YOUR_API_KEY_LASTFM"  # <-- Replace this!
ARTIST_JSON_PATH = r"data\artists\artists_by_genre.json"
OUTPUT_CSV_PATH = r"data\metadata\dataset_meta_v3"
X_SONGS_PER_ARTIST = 5
Y_TOP_SONG_POOL = 20

LASTFM_API_URL = r"http://ws.audioscrobbler.com/2.0/"


# === FUNCTION TO FETCH TOP TRACKS ===
def fetch_top_tracks(artist, limit=Y_TOP_SONG_POOL):
    params = {
        "method": "artist.gettoptracks",
        "artist": artist,
        "api_key": API_KEY,
        "format": "json",
        "limit": limit
    }
    try:
        response = requests.get(LASTFM_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return [track["name"] for track in data["toptracks"]["track"]]
    except Exception as e:
        print(f"Error fetching tracks for {artist}: {e}")
        return []


# === MAIN SCRIPT ===
def main():
    if not os.path.exists(ARTIST_JSON_PATH):
        print(f"Error: {ARTIST_JSON_PATH} not found.")
        return

    with open(ARTIST_JSON_PATH, "r", encoding="utf-8") as f:
        genre_map = json.load(f)

    rows = []
    total_count = 0

    for genre, artists in genre_map.items():
        print(f"Processing genre: {genre}")
        for artist in artists:
            print(f"Fetching songs for artist: {artist}")
            top_tracks = fetch_top_tracks(artist, limit=Y_TOP_SONG_POOL)
            if not top_tracks:
                print(f"Error: Skipping {artist} (no tracks found)")
                continue
            selected_tracks = random.sample(top_tracks, min(X_SONGS_PER_ARTIST, len(top_tracks)))
            for idx, title in enumerate(selected_tracks, start=1):
                print(f"    🎶 [{idx}/{X_SONGS_PER_ARTIST}] {title}")
                rows.append({
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "file_path": ""
                })
                total_count += 1

    with open(OUTPUT_CSV_PATH, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["title", "artist", "genre", "file_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Finished writing {total_count} songs to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
