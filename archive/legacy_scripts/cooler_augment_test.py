import os
import librosa
import numpy as np
from pydub import AudioSegment
import random

# Import your augmentation functions here or define them directly
from scipy.signal import butter, lfilter

# ======== CONFIG ==========
INPUT_FILE = r"C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\demo_songs\all_my_loving_beatles.mp3"  # Change this path
OUTPUT_DIR = r"C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\demo_songs"
SR = 22050
DURATION = 30  # seconds

# ======== AUGMENTATION FUNCTIONS ==========

def pad_or_trim(y, sr, duration):
    target_len = int(duration * sr)
    if len(y) > target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, max(0, target_len - len(y))))

def export_mp3(y, sr, name):
    y = np.clip(y, -1.0, 1.0)
    y_int16 = np.int16(y * 32767)
    audio_segment = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    ).normalize()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, name + ".mp3")
    audio_segment.export(out_path, format="mp3")
    print(f"Saved {out_path}")

def apply_time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def apply_pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def volume_scale(y, scale):
    return y * scale

def low_shelf_filter(y, sr, cutoff, gain_db):
    b, a = butter(2, cutoff / (0.5 * sr), btype='low')
    filtered = lfilter(b, a, y)
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

# ========== MAIN TEST ==========

if __name__ == "__main__":
    print(f"Testing augmentations on {INPUT_FILE}...")

    y, _ = librosa.load(INPUT_FILE, sr=SR, mono=True)
    y = pad_or_trim(y, SR, DURATION)

    export_mp3(y, SR, "original")

    try:
        stretched = apply_time_stretch(y, 0.97)
        stretched = pad_or_trim(stretched, SR, DURATION)
        export_mp3(stretched, SR, "time_stretch_0.97")
    except Exception as e:
        print("Time stretch failed:", e)

    try:
        pitched = apply_pitch_shift(y, SR, 1.0)
        export_mp3(pitched, SR, "pitch_shift_up1")
    except Exception as e:
        print("Pitch shift failed:", e)

    try:
        louder = volume_scale(y, 1.1)
        export_mp3(louder, SR, "volume_up_1.1")
    except Exception as e:
        print("Volume scale failed:", e)

    try:
        filtered = low_shelf_filter(y, SR, cutoff=200.0, gain_db=3.0)
        export_mp3(filtered, SR, "low_shelf_boost")
    except Exception as e:
        print("Low-shelf filter failed:", e)

    try:
        specaug = apply_specaugment(y, SR)
        specaug = pad_or_trim(specaug, SR, DURATION)
        export_mp3(specaug, SR, "specaugment")
    except Exception as e:
        print("SpecAugment failed:", e)

    print("All augmentations complete.")
