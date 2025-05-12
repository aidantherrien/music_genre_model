import requests
import json
import csv
import random
import os


# === CONFIGURATION ===
API_KEY = r"c59114fdaa166a35952e10916ea1e171"  # <-- Replace this!
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
