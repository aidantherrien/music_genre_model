import csv
import os
import random
import subprocess
from pathlib import Path
from yt_dlp import YoutubeDL

# === Constants ===
CSV_PATH = "data\\metadata\\dataset_meta_v3.csv"
RAW_AUDIO_DIR = r"data\raw"
MIN_CLIP_LENGTH = 15
MAX_CLIP_LENGTH = 30
CLIPS_PER_SONG = 2
MAX_SONG_DURATION = 300  # fallback if no duration metadata

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

def download_clips(artist, title, genre, clips_per_song):
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

        clips_data = get_evenly_spaced_clips(duration, MIN_CLIP_LENGTH, MAX_CLIP_LENGTH, clips_per_song)
        if not clips_data:
            print(f"⚠️ Skipping: {artist} - {title} (not enough duration)")
            return None

        try:
            print(f"🎧 Downloading: {url}")
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
    print("🔍 Scanning for existing genre folders...")
    existing_genre_dirs = {folder.name for folder in Path(RAW_AUDIO_DIR).iterdir() if folder.is_dir()}
    print(f"✅ Skipping genres already downloaded: {sorted(existing_genre_dirs)}")

    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            artist, title, genre, file_path = row['artist'], row['title'], row['genre'], row['file_path']
            if file_path.strip() or genre in existing_genre_dirs:
                rows.append(row)
                continue  # Skip already processed or already downloaded genre

            print(f"\n🎵 Processing: {genre} | {artist} - {title}")
            path = download_clips(artist, title, genre, CLIPS_PER_SONG)
            if path:
                row['file_path'] = path
            rows.append(row)

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["title", "artist", "genre", "file_path"])
        writer.writeheader()
        writer.writerows(rows)
    print("\n✅ CSV updated with new file paths.")

if __name__ == "__main__":
    process_csv()
