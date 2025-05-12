import requests
import time
import csv
import os

GENRES = ['rock', 'jazz', 'classical', 'hip hop', 'electronic', 'country', 'metal', 'pop']
ARTISTS_PER_GENRE = 150  # collect more to filter later
SONGS_PER_GENRE = 100
CSV_PATH = r'/csvs/dataset_meta_v1.csv'

HEADERS = {
    'User-Agent': '(projectname)/X.Y (Your Email Address)'
}

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['title', 'artist', 'album', 'year', 'genre', 'recording_mbid', 'file_path'])

    for genre in GENRES:
        print(f"--- {genre.upper()} ---")
        unique_artists = set()
        offset = 0

        # Step 1: Collect unique artists by genre
        while len(unique_artists) < ARTISTS_PER_GENRE:
            params = {
                'query': f'tag:"{genre}"',
                'fmt': 'json',
                'limit': 100,
                'offset': offset
            }
            res = requests.get("https://musicbrainz.org/ws/2/artist/", params=params, headers=HEADERS)
            data = res.json()
            for artist in data.get('artists', []):
                if 'name' in artist and artist['name'] not in unique_artists:
                    unique_artists.add(artist['name'])
            offset += 100
            time.sleep(1)
            if not data.get('artists'):
                break

        print(f"Found {len(unique_artists)} artists tagged {genre}")

        collected = 0
        for artist_name in list(unique_artists):
            if collected >= SONGS_PER_GENRE:
                break

            # Step 2: Get recordings by this artist
            params = {
                'query': f'artist:"{artist_name}"',
                'fmt': 'json',
                'limit': 5
            }
            res = requests.get("https://musicbrainz.org/ws/2/recording/", params=params, headers=HEADERS)
            recordings = res.json().get('recordings', [])
            if not recordings:
                continue

            # Step 3: Try to extract one valid recording with metadata
            for rec in recordings:
                title = rec.get('title', '')
                mbid = rec.get('id', '')
                ac = rec.get('artist-credit', [])
                artist = ac[0]['name'] if ac else ''
                releases = rec.get('releases', [])
                if not releases:
                    continue
                album = releases[0]['title']
                date = releases[0].get('date', '')
                year = date.split('-')[0] if date else ''
                if not year or int(year) < 1950 or int(year) > 2024:
                    continue  # keep only realistic modern music
                writer.writerow([title, artist, album, year, genre, mbid, ''])
                collected += 1
                break  # only take one recording per artist
            time.sleep(1)

        print(f"{collected} songs collected for {genre}")
