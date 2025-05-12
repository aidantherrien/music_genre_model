import os
import random
import librosa
import numpy as np
from pydub import AudioSegment


# ============================================================
# MUSIC DATA AUGMENTATION SCRIPT
# ============================================================
# This script performs audio augmentation on a set of music 
# files to artificially increase the size of the dataset for 
# machine learning tasks, such as genre classification. It 
# applies various transformations to the audio files, including 
# time stretching, pitch shifting, volume scaling, low shelf 
# filtering, and mel spectrogram augmentation.
# 
# The augmented audio files are then saved in the specified 
# output directory, maintaining the original genre folder 
# structure. This script supports MP3, WAV, FLAC, and OGG file 
# formats and processes all audio files under the provided 
# input directory.
# 
# ========================= CONFIGURABLE CONSTANTS =========================
# INPUT_DIR       : Directory containing raw audio files, organized by genre.
# OUTPUT_DIR      : Directory where the augmented audio files will be saved.
# AUGS_PER_FILE   : Number of augmented versions to generate per original file.
# TARGET_SR       : Target sample rate for the audio files (default 22050 Hz).
# AUDIO_DURATION  : Duration in seconds to which each audio file is padded or trimmed.
# VOLUME_SCALE_RANGE : Range for random volume scaling (values between 0.85 and 1.15).
# PITCH_SHIFT_RANGE : Range for random pitch shifting in semitones (-1.5 to +1.5).
# TIME_STRETCH_RANGE : Range for time stretching (values between 0.95 and 1.05).
# LOW_SHELF_CUTOFF_RANGE : Range for low-shelf filter cutoff frequencies (100 Hz to 300 Hz).
# LOW_SHELF_GAIN_DB_RANGE : Range for low-shelf filter gain in dB (-4.0 to +4.0).
# MEL_SPEC_AUGMENT : Boolean flag to apply mel spectrogram augmentation (default: True).
# 
# ========================= AUGMENTATION FUNCTIONS =========================
# Functions defined here apply various augmentations to the input audio:
# - apply_time_stretch: Stretches the audio in time.
# - apply_pitch_shift: Shifts the pitch of the audio.
# - volume_scale: Scales the audio volume randomly within the specified range.
# - low_shelf_filter: Applies a low-shelf filter to modify low-frequency content.
# - apply_specaugment: Applies random masking to mel spectrogram features.
# - pad_or_trim: Pads or trims the audio to a fixed duration.
# - export_as_mp3: Exports the augmented audio as an MP3 file.
# 
# ========================= MAIN AUGMENTATION ==============================
# The augment_file function loads an audio file, applies augmentations, 
# and saves the augmented audio files into the output directory.
# The augment_all function traverses the input directory, processes 
# all audio files, and generates augmented files.
# 
# ============================================================


# ======================= CONFIGURABLE CONSTANTS ========================
INPUT_DIR = r"data\raw"
OUTPUT_DIR = r"data\augmented"
AUGS_PER_FILE = 2
TARGET_SR = 22050
AUDIO_DURATION = 30
VOLUME_SCALE_RANGE = (0.85, 1.15)
PITCH_SHIFT_RANGE = (-1.5, 1.5)
TIME_STRETCH_RANGE = (0.95, 1.05)
LOW_SHELF_CUTOFF_RANGE = (100.0, 300.0)
LOW_SHELF_GAIN_DB_RANGE = (-4.0, 4.0)
MEL_SPEC_AUGMENT = True

# ========================= AUGMENTATION FUNCTIONS =========================

def apply_time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

def apply_pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def volume_scale(audio, scale):
    return audio * scale

def low_shelf_filter(audio, sr, cutoff, gain_db):
    from scipy.signal import butter, lfilter
    b, a = butter(2, cutoff / (0.5 * sr), btype='low')
    filtered = lfilter(b, a, audio)
    gain = 10 ** (gain_db / 20)
    return filtered * gain

def apply_specaugment(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    f = random.randint(0, 20)
    f0 = random.randint(0, mel_db.shape[0] - f)
    mel_db[f0:f0+f, :] = 0
    t = random.randint(0, 40)
    t0 = random.randint(0, mel_db.shape[1] - t)
    mel_db[:, t0:t0+t] = 0
    return librosa.feature.inverse.mel_to_audio(librosa.db_to_power(mel_db), sr=sr)

def pad_or_trim(y, sr, duration):
    target_len = int(duration * sr)
    if len(y) > target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, max(0, target_len - len(y))))

def export_as_mp3(y, sr, output_path):
    # Convert to int16 and wrap as AudioSegment
    y_int16 = np.int16(y * 32767)
    audio_segment = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    ).normalize()
    audio_segment.export(output_path, format="mp3")

# ========================= MAIN AUGMENTATION ==============================

def augment_file(file_path, output_folder, sr=TARGET_SR):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        y = pad_or_trim(y, sr, AUDIO_DURATION)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        genre = os.path.basename(os.path.dirname(file_path))
        print(f"Augmenting {genre}/{base_name}...")

        for i in range(AUGS_PER_FILE):
            aug_y = y.copy()

            if random.random() < 0.9:
                aug_y = apply_time_stretch(aug_y, random.uniform(*TIME_STRETCH_RANGE))
            if random.random() < 0.9:
                aug_y = apply_pitch_shift(aug_y, sr, random.uniform(*PITCH_SHIFT_RANGE))
            if random.random() < 0.9:
                aug_y = volume_scale(aug_y, random.uniform(*VOLUME_SCALE_RANGE))
            if random.random() < 0.5:
                aug_y = low_shelf_filter(
                    aug_y,
                    sr,
                    cutoff=random.uniform(*LOW_SHELF_CUTOFF_RANGE),
                    gain_db=random.uniform(*LOW_SHELF_GAIN_DB_RANGE)
                )
            if MEL_SPEC_AUGMENT and random.random() < 0.5:
                aug_y = apply_specaugment(aug_y, sr)

            # Prepare output
            genre_out_dir = os.path.join(output_folder, genre)
            os.makedirs(genre_out_dir, exist_ok=True)
            out_file = os.path.join(genre_out_dir, f"{base_name}_aug{i+1}.mp3")

            export_as_mp3(aug_y, sr, out_file)
            print(f"  Saved: {out_file}")

    except Exception as e:
        print(f"  Failed to augment {file_path}: {e}")

# ============================= SCRIPT ENTRY ===============================

def augment_all(input_root, output_root):
    total_files = 0
    for genre_folder, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                full_path = os.path.join(genre_folder, f)
                augment_file(full_path, output_root)
                total_files += 1
    print(f"Processed {total_files} original files.")

if __name__ == "__main__":
    print(f"Starting augmentation from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    augment_all(INPUT_DIR, OUTPUT_DIR)
    print("Augmentation complete.")
