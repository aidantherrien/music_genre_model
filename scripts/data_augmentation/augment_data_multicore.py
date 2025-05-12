import os
import random
import librosa
import multiprocessing
import numpy as np
from pydub import AudioSegment


# ==============================================================================
# MULTICORE AUDIO AUGMENTATION SCRIPT
# ==============================================================================
# This script applies several audio augmentation techniques to a collection of 
# audio files in a given directory. The augmented files are saved in an output 
# directory, organized by genre.
#
# This script is functionally the same as augment_data.py, except it allows for
# multicore processing to get the job dont faster.
#


# ======================= CONFIGURABLE CONSTANTS ========================
INPUT_DIR = r"data\raw"
OUTPUT_DIR = r"data\augmented"
AUGS_PER_FILE = 2
TARGET_SR = 22050
AUDIO_DURATION = 30  # in seconds
VOLUME_SCALE_RANGE = (0.85, 1.15)
PITCH_SHIFT_RANGE = (-1.5, 1.5)
TIME_STRETCH_RANGE = (0.95, 1.05)
LOW_SHELF_CUTOFF_RANGE = (100.0, 300.0)
LOW_SHELF_GAIN_DB_RANGE = (-3.0, 3.0)
MEL_SPEC_AUGMENT = False                         # really doesn't work
CORES = 6                                        # of cores you wish to use

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
    f0 = random.randint(0, max(1, mel_db.shape[0] - f))
    mel_db[f0:f0+f, :] = 0
    t = random.randint(0, 40)
    t0 = random.randint(0, max(1, mel_db.shape[1] - t))
    mel_db[:, t0:t0+t] = 0
    return librosa.feature.inverse.mel_to_audio(librosa.db_to_power(mel_db), sr=sr)

def pad_or_trim(y, sr, duration):
    target_len = int(duration * sr)
    if len(y) > target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, max(0, target_len - len(y))))

def export_as_mp3(y, sr, output_path):
    y = np.clip(y, -1.0, 1.0)  # Ensure no overflow before conversion
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

            try:
                if random.random() < 0.9:
                    stretch_rate = random.uniform(*TIME_STRETCH_RANGE)
                    aug_y = apply_time_stretch(aug_y, stretch_rate)
                    aug_y = pad_or_trim(aug_y, sr, AUDIO_DURATION)

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

                aug_y = pad_or_trim(aug_y, sr, AUDIO_DURATION)  # Final length fix
                aug_y = np.nan_to_num(aug_y)  # Replace any NaNs or Infs

                genre_out_dir = os.path.join(output_folder, genre)
                os.makedirs(genre_out_dir, exist_ok=True)
                out_file = os.path.join(genre_out_dir, f"{base_name}_aug{i+1}.mp3")

                export_as_mp3(aug_y, sr, out_file)
                print(f"  Saved: {out_file}")

            except Exception as e:
                print(f"  Augmentation step failed for {file_path}: {e}")

    except Exception as e:
        print(f"  Failed to augment {file_path}: {e}")

# ============================= SCRIPT ENTRY ===============================

def get_all_audio_files(input_root):
    audio_files = []
    for genre_folder, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                full_path = os.path.join(genre_folder, f)
                audio_files.append(full_path)
    return audio_files

def augment_all_parallel(input_root, output_root):
    audio_files = get_all_audio_files(input_root)
    total_files = len(audio_files)
    print(f"Discovered {total_files} files for augmentation...")

    num_workers = max(1, int(CORES))

    print(f"Using {num_workers} worker processes...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(augment_file, [(path, output_root) for path in audio_files])

    print(f"Processed {total_files} original files.")

if __name__ == "__main__":
    print(f"Starting augmentation from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    augment_all_parallel(INPUT_DIR, OUTPUT_DIR)
    print("Augmentation complete.")
