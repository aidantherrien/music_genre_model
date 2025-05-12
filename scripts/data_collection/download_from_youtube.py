import csv
import os
import random
import subprocess
from pathlib import Path
from yt_dlp import YoutubeDL

# === Song Clip Downloader and Processor (download_from_youtube.py) ===
# This script processes a CSV file containing metadata for songs (artist, title, genre),
# searches YouTube for each song, and downloads audio clips based on specified durations.
# The clips are then saved in a genre-specific directory, and the CSV is updated with 
# the file paths of the downloaded clips.
#
# === Constants ===
# CSV_PATH: Path to the CSV file with song metadata
# RAW_AUDIO_DIR: Directory to store raw audio files
# MIN_CLIP_LENGTH: Minimum length for each audio clip (in seconds)
# MAX_CLIP_LENGTH: Maximum length for each audio clip (in seconds)
# CLIPS_PER_SONG: Number of clips to extract from each song
# MAX_SONG_DURATION: Maximum allowed song duration for processing (fallback value)
#
# === Functions ===
# - get_evenly_spaced_clips: Generates evenly spaced audio clips from the full song duration
# - sanitize_filename: Sanitizes the song title to create a valid filename
# - download_clips: Searches for the song on YouTube, downloads the audio, splits it into clips, and returns the file paths
# - process_csv: Reads the CSV file, processes each song, downloads clips, and updates the CSV with file paths


# === Constants ===
CSV_PATH = r"data\metadata\dataset_meta_v3.2.csv"
RAW_AUDIO_DIR = r"data\raw"
MIN_CLIP_LENGTH = 15
MAX_CLIP_LENGTH = 30
CLIPS_PER_SONG = 2
MAX_SONG_DURATION = 300  # in seconds, fallback in case we can't get exact


# === Utilities ===

def get_evenly_spaced_clips(song_duration, min_len, max_len, clips_per_song):
    clip_length = random.randint(min_len, max_len)
    total_required = clip_length * clips_per_song

    if total_required > song_duration:
        clips_per_song = max(1, song_duration // clip_length)
        if clips_per_song == 0:
            return []
        total_required = clip_length * clips_per_song

    spacing = (song_duration - total_required) // (clips_per_song + 1)
    current_time = spacing

    clips = []
    for _ in range(clips_per_song):
        start = current_time
        end = start + clip_length
        if end > song_duration:
            break
        clips.append((start, end))
        current_time = end + spacing
    return clips

def sanitize_filename(name):
    return name.lower().replace(" ", "_").replace("/", "_")

def download_clips(artist, title, genre, clips):
    query = f"ytsearch1:{artist} {title}"
    save_dir = os.path.join(RAW_AUDIO_DIR, genre)
    os.makedirs(save_dir, exist_ok=True)

    base_filename = sanitize_filename(f"{artist}_{title}")
    output_template = os.path.join(save_dir, f"{base_filename}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with YoutubeDL(ydl_opts) as ydl:
        print(f"🔍 Searching YouTube for: {artist} - {title}")
        try:
            result = ydl.extract_info(query, download=False)['entries'][0]
            url = result['webpage_url']
            duration = result.get("duration", MAX_SONG_DURATION)
        except Exception as e:
            print(f"Failed to fetch: {artist} - {title}: {e}")
            return None

        clips_data = get_evenly_spaced_clips(duration, MIN_CLIP_LENGTH, MAX_CLIP_LENGTH, CLIPS_PER_SONG)
        if not clips_data:
            print(f"Skipping: {artist} - {title} (not enough duration)")
            return None

        try:
            print(f"Downloading: {url}")
            ydl.download([url])
        except Exception as e:
            print(f"Download failed: {e}")
            return None

        input_file = os.path.join(save_dir, f"{base_filename}.mp3")
        file_paths = []

        for i, (start, end) in enumerate(clips_data):
            output_path = os.path.join(save_dir, f"{base_filename}_clip{i+1}.mp3")
            cmd = [
                "ffmpeg", "-y", "-i", input_file,
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            file_paths.append(output_path)

        # Clean up original full MP3
        if os.path.exists(input_file):
            os.remove(input_file)

        return file_paths[0] if file_paths else None

# === Main Logic ===

def process_csv():
    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            artist, title, genre, file_path = row['artist'], row['title'], row['genre'], row['file_path']
            if file_path.strip():
                continue  # already processed

            print(f"Processing: {genre} | {artist} - {title}")
            path = download_clips(artist, title, genre, CLIPS_PER_SONG)
            if path:
                row['file_path'] = path
            rows.append(row)

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["title", "artist", "genre", "file_path"])
        writer.writeheader()
        writer.writerows(rows)
    print("CSV updated with new file paths.")

if __name__ == "__main__":
    process_csv()
