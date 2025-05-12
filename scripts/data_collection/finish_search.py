import requests
import json
import csv
import os
import time
import random

# === CONFIGURATION ===
API_KEY = r"c59114fdaa166a35952e10916ea1e171"  # Replace with your API key
LASTFM_API_URL = r"http://ws.audioscrobbler.com/2.0/"
ARTIST_JSON_PATH = r"data\artists\artists_by_genre.json"  # Path to your artist JSON file
OUTPUT_CSV_PATH = r"data\metadata\dataset_meta_v3.2"  # Path where you want the CSV output
X_SONGS_PER_ARTIST = 5
Y_TOP_SONG_POOL = 20

# List of genres to try and pull
GENRES_TO_TRY = [
    "classic_rock",
    "alternative_rock",
    "alternative",
    "pop_punk",
    "funk",
    "pop_country",
    "fusion"
]


with open(ARTIST_JSON_PATH, "r", encoding="utf-8") as f:
    genre_map = json.load(f)

print(genre_map.keys())  # This will list all available genres in the JSON file.


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

        # Check if 'toptracks' is in the response
        if "toptracks" not in data or not data["toptracks"].get("track"):
            print(f"⚠️  No tracks for {artist}")
            return []
        return [track["name"] for track in data["toptracks"]["track"]]
    except Exception as e:
        print(f"Error fetching tracks for {artist}: {e}")
        return []

# === FUNCTION TO LOAD EXISTING CSV ===
def load_existing_csv(csv_path):
    existing_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                genre = row["genre"]
                if genre not in existing_data:
                    existing_data[genre] = set()
                existing_data[genre].add(row["artist"])
    return existing_data

# === MAIN SCRIPT ===
def main():
    if not os.path.exists(ARTIST_JSON_PATH):
        print(f"Error: {ARTIST_JSON_PATH} not found.")
        return

    # Read artists by genre
    with open(ARTIST_JSON_PATH, "r", encoding="utf-8") as f:
        genre_map = json.load(f)

    # Load existing CSV data to check which genres are already populated
    existing_data = load_existing_csv(OUTPUT_CSV_PATH)

    rows = []
    total_count = 0
    failed_artists = {}

    # Try fetching tracks for all artists in selected genres
    for genre in GENRES_TO_TRY:
        if genre in existing_data:
            print(f"Skipping genre: {genre} (already populated)")
            continue

        if genre not in genre_map:
            print(f"Genre {genre} not found in the artist JSON file.")
            continue

        artists = genre_map[genre]
        print(f"Processing genre: {genre}")
        for artist in artists:
            print(f"Fetching songs for artist: {artist}")
            top_tracks = fetch_top_tracks(artist, limit=Y_TOP_SONG_POOL)
            if not top_tracks:
                if genre not in failed_artists:
                    failed_artists[genre] = []
                failed_artists[genre].append(artist)
            else:
                selected_tracks = random.sample(top_tracks, min(X_SONGS_PER_ARTIST, len(top_tracks)))
                for idx, title in enumerate(selected_tracks, start=1):
                    rows.append({
                        "title": title,
                        "artist": artist,
                        "genre": genre,
                        "file_path": ""
                    })
                    total_count += 1

    # If there are failed genres/artists, try fetching them again
    if failed_artists:
        print("\nRetrying failed genres...")
        for genre, artists in failed_artists.items():
            print(f"Retrying genre: {genre}")
            for artist in artists:
                print(f"Fetching songs for artist: {artist} (Retrying)")
                time.sleep(5)  # Sleep to avoid rate-limiting
                top_tracks = fetch_top_tracks(artist, limit=Y_TOP_SONG_POOL)
                if top_tracks:
                    selected_tracks = random.sample(top_tracks, min(X_SONGS_PER_ARTIST, len(top_tracks)))
                    for idx, title in enumerate(selected_tracks, start=1):
                        rows.append({
                            "title": title,
                            "artist": artist,
                            "genre": genre,
                            "file_path": ""
                        })
                        total_count += 1
                else:
                    print(f"Error: Skipping {artist} again (no tracks found)")

    # Append the results to the CSV file
    if rows:
        with open(OUTPUT_CSV_PATH, "a", newline='', encoding="utf-8") as csvfile:
            fieldnames = ["title", "artist", "genre", "file_path"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # If the file is empty, write the header first
            if os.stat(OUTPUT_CSV_PATH).st_size == 0:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)

        print(f"Finished writing {total_count} songs to {OUTPUT_CSV_PATH}")
    else:
        print("No new songs to add.")

if __name__ == "__main__":
    main()
