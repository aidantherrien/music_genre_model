import librosa

path = "/data/archive/Data/genres_original/blues/blues.00000.wav"
y, sr = librosa.load(path, duration=30)
print(f"Audio length: {len(y)}, Sampling rate: {sr}")
