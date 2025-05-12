import os
import csv
import requests
from bs4 import BeautifulSoup

# Paths and CSV configuration
CSV_PATH = "/notebooks"
SONG_LIST_CSV = os.path.join(CSV_PATH, "historical_songs.csv")

# Ensure directory exists
os.makedirs(CSV_PATH, exist_ok=True)

# Wikipedia URL (Rolling Stone's 500 Greatest Songs of All Time)
WIKI_URL = "https://en.wikipedia.org/wiki/Rolling_Stone's_500_Greatest_Songs_of_All_Time"


def scrape_wikipedia_song_list(url):
    """Scrapes a Wikipedia page for song titles and artists."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    songs = []

    # Scrape song titles and artists from the table
    for row in soup.select("table.wikitable tbody tr")[1:]:  # Skip the header row
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
        writer.writerow(["Song Title"])  # Header row
        for song in song_list:
            writer.writerow([song])  # Each song in a new row


# Scrape song list and save to CSV
historical_songs = scrape_wikipedia_song_list(WIKI_URL)

for song in historical_songs:
    print(song)

if historical_songs:
    print(f"Found {len(historical_songs)} songs. Saving to CSV...")
    save_song_list_to_csv(historical_songs)
else:
    print("No songs were found.")
