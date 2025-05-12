import musicbrainzngs
import random
import os
import csv
import time

# Setup the MusicBrainz user agent
musicbrainzngs.set_useragent("MusicGenreCollector", "3.0", "aidanmtherrien@gmail.com")

# Constants
TOTAL_SONGS_PER_GENRE = 50  # Total number of songs per genre
GENRE_LIST = ['rock', 'jazz', 'blues', 'classical', 'pop', 'country', 'metal', 'hip-hop']
ARTISTS_PER_GENRE = 25  # Number of artists per genre
SONGS_PER_ARTIST = 2  # Number of songs per artist
RETRIES = 3  # Number of retry attempts for failed queries

# CSV Output file
OUTPUT_CSV = r"C:\Users\aidan\PycharmProjects\pythonProject19\csvs\dataset_meta_v2.csv"

# Manually input the list of artists per genre
artists_by_genre = {
    'rock': ['Elvis Presley', 'The Mountain Goats', 'Sting', 'Canned Heat', 'The Cure', 'The Beatles', 'Led Zeppelin', 'Queen', 'AC/DC', 'Nirvana', 'Pearl Jam', 'The Rolling Stones', 'Pink Floyd', 'The Doors', 'Bob Dylan', 'The Clash', 'U2', 'Radiohead', 'The Who', 'The Ramones', 'Jimi Hendrix', 'David Bowie', 'The Kinks', 'The Smiths', 'The Police'],
    'jazz': ['Miles Davis', 'John Coltrane', 'Louis Armstrong', 'Duke Ellington', 'Charlie Parker', 'Thelonious Monk', 'Chet Baker', 'Billie Holiday', 'Ella Fitzgerald', 'Count Basie', 'Stan Getz', 'Charles Mingus', 'Art Blakey', 'Wayne Shorter', 'Herbie Hancock', 'Wynton Marsalis', 'Max Roach', 'Dizzy Gillespie', 'Oscar Peterson', 'Dave Brubeck', 'Keith Jarrett', 'Sarah Vaughan', 'Benny Goodman', 'Johnny Mercer', 'Pat Metheny'],
    'blues': ['B.B. King', 'Muddy Waters', 'Robert Johnson', 'Howlin\' Wolf', 'Willie Dixon', 'John Lee Hooker', 'Stevie Ray Vaughan', 'Buddy Guy', 'Etta James', 'Elmore James', 'Sonny Boy Williamson', 'T-Bone Walker', 'Albert King', 'Little Walter', 'Jimmy Reed', 'Koko Taylor', 'Muddy Waters', 'Freddie King', 'Otis Rush', 'John Mayall', 'Big Mama Thornton', 'Sam Myers', 'Johnny Winter', 'Blind Willie McTell', 'Popa Chubby', 'Peter Green'],
    'classical': ['Ludwig van Beethoven', 'Wolfgang Amadeus Mozart', 'Johann Sebastian Bach', 'Pyotr Ilyich Tchaikovsky', 'Frédéric Chopin', 'Johannes Brahms', 'Claude Debussy', 'Johann Strauss II', 'Antonio Vivaldi', 'Giuseppe Verdi', 'Gustav Mahler', 'Sergei Rachmaninoff', 'Richard Wagner', 'Joseph Haydn', 'Franz Schubert', 'Felix Mendelssohn', 'Samuel Barber', 'Edward Elgar', 'Maurice Ravel', 'Igor Stravinsky', 'Camille Saint-Saëns', 'Dmitri Shostakovich', 'Carl Orff', 'Leonard Bernstein', 'Aaron Copland'],
    'pop': ['Ariana Grande', 'Taylor Swift', 'Ed Sheeran', 'Dua Lipa', 'Justin Bieber', 'Billie Eilish', 'Shawn Mendes', 'Lady Gaga', 'Katy Perry', 'Bruno Mars', 'Post Malone', 'Beyoncé', 'The Weeknd', 'Harry Styles', 'Rihanna', 'Kendrick Lamar', 'Drake', 'Sia', 'Maroon 5', 'Selena Gomez', 'Sam Smith', 'Shakira', 'Adele', 'Miley Cyrus', 'Lizzo', 'BTS'],
    'country': ['Johnny Cash', 'Dolly Parton', 'Garth Brooks', 'Willie Nelson', 'George Strait', 'Reba McEntire', 'Carrie Underwood', 'Tim McGraw', 'Alan Jackson', 'Loretta Lynn', 'Luke Bryan', 'Blake Shelton', 'Miranda Lambert', 'Brad Paisley', 'Keith Urban', 'Shania Twain', 'Hank Williams', 'Kenny Rogers', 'Chris Stapleton', 'George Jones', 'Conway Twitty', 'Tammy Wynette', 'Vince Gill', 'Faith Hill', 'Travis Tritt', 'Jason Aldean'],
    'metal': ['Metallica', 'Black Sabbath', 'Iron Maiden', 'Judas Priest', 'Slayer', 'Megadeth', 'Anthrax', 'Pantera', 'System of a Down', 'Slipknot', 'Tool', 'Opeth', 'Lamb of God', 'Amon Amarth', 'Korn', 'Death', 'Alice in Chains', 'Motorhead', 'Dimmu Borgir', 'Behemoth', 'Children of Bodom', 'Testament', 'Ghost', 'Avenged Sevenfold', 'Cradle of Filth', 'Rammstein'],
    'hip-hop': ['Tupac Shakur', 'The Notorious B.I.G.', 'Jay-Z', 'Nas', 'Kendrick Lamar', 'Drake', 'Eminem', 'Lil Wayne', 'Kanye West', 'Snoop Dogg', 'Rakim', 'A Tribe Called Quest', 'OutKast', 'Missy Elliott', 'Run-D.M.C.', 'Public Enemy', 'Wu-Tang Clan', 'KRS-One', 'Ice Cube', 'Lil Nas X', 'Travis Scott', 'J. Cole', 'Cardi B', 'Megan Thee Stallion', 'Tyler, The Creator', 'Dr. Dre']
}

# Functions to interact with MusicBrainz
def get_artist_mbid(artist_name):
    """Retrieve the MBID (MusicBrainz Identifier) for an artist."""
    result = musicbrainzngs.search_artists(artist_name, limit=1)
    if result['artist-list']:
        return result['artist-list'][0]['id']
    return None

def get_release_groups(mbid, retries=RETRIES):
    """Retrieve release groups (e.g., 'Greatest Hits') for a given artist MBID."""
    attempt = 0
    while attempt < retries:
        try:
            result = musicbrainzngs.get_artist_by_id(mbid, includes=['release-groups'])
            release_groups = result['artist']['release-group-list']
            # Filter out 'best of', 'greatest hits', and 'compilations'
            return [group for group in release_groups if 'greatest hits' in group['title'].lower() or 'best of' in group['title'].lower()]
        except Exception as e:
            attempt += 1
            print(f"Error retrieving release groups for MBID {mbid}: {e}. Retrying ({attempt}/{retries})...")
            time.sleep(1)
    return []

def get_tracks_from_release_group(release_group_id):
    """Retrieve tracks from a specific release group (album collection)."""
    try:
        # Use 'recordings' instead of 'track-list' to get detailed track information
        release_group = musicbrainzngs.get_release_group_by_id(release_group_id, includes=['releases'])
        tracks = []
        for release in release_group['release-group']['release-list']:
            release_id = release['id']
            release_info = musicbrainzngs.get_release_by_id(release_id, includes=['recordings'])
            tracks.extend([
                (track['title'], release['title'], release['date'], track['recording']['id'])
                for track in release_info['release']['recording-list'] if 'recording' in track
            ])
        return tracks
    except Exception as e:
        print(f"Error retrieving tracks from release group {release_group_id}: {e}")
        return []


def collect_songs_for_genre(genre, artists, num_songs=50):
    collected = []
    artist_song_count = {}

    for artist in artists:
        if len(collected) >= num_songs:
            break

        mbid = get_artist_mbid(artist)
        if not mbid:
            continue

        release_groups = get_release_groups(mbid)
        if not release_groups:
            print(f"Skipping {artist} - no popular releases found.")
            continue

        print(f"→ Scanning release groups for artist: {artist}")
        random.shuffle(release_groups)

        for group in release_groups:
            tracks = get_tracks_from_release_group(group['id'])
            if not tracks:
                continue

            available_slots = SONGS_PER_ARTIST - artist_song_count.get(artist, 0)
            selected = random.sample(tracks, min(len(tracks), available_slots))

            for title, album, year, recording_mbid in selected:
                entry = {
                    'title': title,
                    'artist': artist,
                    'album': album,
                    'year': year,
                    'genre': genre,
                    'recording_mbid': recording_mbid,
                    'file_path': f"C:/Users/aidan/PycharmProjects/pythonProject19/current_data/{genre}/{artist} - {title}.mp3"  # Change to your actual file structure
                }
                collected.append(entry)
                artist_song_count[artist] = artist_song_count.get(artist, 0) + 1
                print(f"[{len(collected):03}/{num_songs}] Saved: \"{title}\" by {artist} ({album}, {year})")

            if len(collected) >= num_songs:
                break

    return collected

def save_songs_to_csv(songs):
    """Save collected songs to a CSV file."""
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'artist', 'album', 'year', 'genre', 'recording_mbid', 'file_path'])
        writer.writeheader()
        for song in songs:
            writer.writerow(song)

def main():
    all_songs = []
    for genre, artists in artists_by_genre.items():
        print(f"\nStarting genre: {genre}")
        songs = collect_songs_for_genre(genre, artists[:ARTISTS_PER_GENRE], num_songs=TOTAL_SONGS_PER_GENRE)
        all_songs.extend(songs)

    save_songs_to_csv(all_songs)
    print("Finished collecting and saving songs.")

if __name__ == "__main__":
    main()
