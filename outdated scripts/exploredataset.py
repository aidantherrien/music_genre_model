import os
import librosa.display
import matplotlib.pyplot as plt

# Load and inspect data
data_path = "/data/archive/Data/genres_original"
genres = os.listdir(data_path)

print("Genres:", genres)
for genre in genres:
    files = os.listdir(os.path.join(data_path, genre))
    print(f"{genre}: {len(files)} files")

# Visualize data for a sample
file_path = "/data/archive/Data/genres_original/jazz/jazz.00000.wav"
y, sr = librosa.load(file_path, duration=30)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of a Jazz Song")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
