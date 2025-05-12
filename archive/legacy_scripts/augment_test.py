import librosa
from pydub import AudioSegment
import numpy as np

# Constants for the test
TARGET_SR = 22050  # Sample rate
AUDIO_FILE_PATH = r'C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\demo_songs\naima_john_coltrane.mp3'  # Path to the audio file
OUTPUT_FILE_PATH = r'C:\Users\aidan\Documents\VSCode Projects\music_genre_model\data\demo_songs\naima_aug.mp3'  # Output file path

# Define the augmentation function
def apply_time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

# Load an audio file (keep the original stereo, mono=True to avoid pitch shifts)
y, sr = librosa.load(AUDIO_FILE_PATH, sr=TARGET_SR, mono=True)

# Apply time stretching with a rate of 0.95 (slightly slower, without changing pitch)
augmented_audio = apply_time_stretch(y, 0.95)

# Ensure the audio is in int16 format for proper conversion to AudioSegment
augmented_audio_int16 = np.int16(augmented_audio * 32767)  # Scale to int16 range

# Convert the NumPy array back to a pydub AudioSegment (for mono handling)
augmented_audio_segment = AudioSegment(
    augmented_audio_int16.tobytes(), 
    frame_rate=sr,
    sample_width=2,  # 2 bytes per sample for int16
    channels=1  # Mono audio
)

# Normalize audio before exporting
augmented_audio_segment = augmented_audio_segment.normalize()

# Export the augmented audio as MP3
augmented_audio_segment.export(OUTPUT_FILE_PATH, format="mp3")

print(f"Test audio saved to {OUTPUT_FILE_PATH}")
