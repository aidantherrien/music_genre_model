## music_genre_model
A Python-based AI model that classifies music genres by analyzing audio features. It uses the Last.fm API to pull songs across a variety of genres. It stores genre, title, and artist information in a CSV file for data collection purposes. Audio features such as MFCC, Chroma, and Spectral Contrast are extracted from the audio files and used for model training. Songs are downloaded from YouTube.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Data](#data)
- [Scripts](#scripts)
- [Notebooks](#notebooks)
- [Notes](#notes)

# Features
This model classifies music into genres based on extracted audio features. The main purpose of this project is to classify songs using deep learning models, while experimenting with audio processing techniques like MFCC and Chroma.

Core Functionalities:
  - Creating a genre classification AI model using TensorFlow with a CNN-LSTM hybrid architecture.
  - Collection of training data using the Last.fm API, YouTube, and audio feature extraction tools like Librosa and sklearn.
  - Data augmentation in order to increase size of data set artificially.
  - Storing metadata (genre, title, artist) in CSV format for easy access and training.
  - Extracting audio features (MFCC, Chroma, Spectral Contrast) from MP3 files and storing them in NPZ format.
  - Training and saving the AI model in a .keras format for easy deployment with TensorFlow.

Future direction of the project:
  - Add more data, struggling to generalize
  - Consider that some genres are too similar, leading to overlap
  - Potential (RNN) Recursive Neural Network implementation. 
  - Utilize the full capacity of the Last.fm API to enhance metadata retrieval and genre definitions.
  - Use PostgreSQL to manage a larger database of songs over time.

## Installation
This project has a few prerequisites:
  - Python v3.11
  - FFmpeg (required for YouTube downloads)

Installing FFmpeg:
  - For Windows users, download from the FFmpeg website and add it to your system path.
  - Linux/MacOS users may use one of the following commands:

Commands:
  - sudo apt install ffmpeg # Debian/Ubuntu
  - brew install ffmpeg # macOS

Cloning Repository:
  - git clone https://github.com/aidantherrien/music_genre_model.git
  - cd music_genre_model

Installing Dependencies:
  - Other dependencies are listed in the requirements.txt file. To install them, run the following command:
  - pip install -r requirements.txt

## Setup
Follow the following steps before first using the program:

Create Directories:
data/: Folder to store audio files, CSV, and NPZ files.
current_data/: Temporary storage for downloaded MP3 files.
Configure API Keys: Place your API keys for the Last.fm API in the config.py file or set them as environment variables.
Update .gitignore: Ensure sensitive files and large files like audio data are ignored (e.g., data/, *.mp3, etc.).
Adjust config.py: Set paths and model parameters, including mappings for genre-to-neuron.

## Data
Audio Data: Songs are downloaded from YouTube using metadata (title, artist, genre) retrieved from the Last.fm API.
Metadata: The metadata is stored in a CSV file with columns such as title, artist, album, year, genre, and recording_mbid.
Audio Features: Audio features (MFCC, Chroma, Spectral Contrast) are extracted using Librosa and stored in NPZ format for model training.

Data Pipeline:
- Data Collection: Downloads music and stores them in genre-specific folders within current_data/.
- Data Augmentation: Various audio augmentation techniques are used, including pitch shifting, tempo shifting, noise, and more to - increase dataset size.
- Feature Extraction: Extracts audio features for each song, normalizes them, and saves them in both CSV and NPZ formats.
- Training Data: Preprocessed feature vectors (flattened) are used for training the genre classification model.

## Scripts
- get_songs.py: Automates the process of collecting metadata from the Last.fm API and saving metadata to CSV
- download_from_youtube.py: Moves through metadata CSV and downloads 2 random 30 second clips of each song (adjustable values)
- augment_data.py: Performs data augmentation of the dataset, and stores the results in a seperate folder, parameters of augmentation can be edited.
- encode_vectors.py: Extracts audio features (MFCC, Chroma, Spectral Contrast, etc) from MP3 files and saves them to CSV and NPZ formats as flattened feature vectors for training.
- train_h5.py: Trains the CNN-LSTM hybrid model on the extracted features.
- train_keras.py: Trains the old MLP models on the extracted features.

## Notebooks
Several Jupyter Notebooks are included in the repository to assist with dataset exploration, model testing, and prediction.

These notebooks are ideal for:
  - Visualizing feature distributions (e.g., MFCC, Chroma, Spectral Contrast) to better understand the dataset.
  - Loading and testing the trained model on new songs for real-time genre prediction.
  - Interpreting prediction confidence using bar charts and softmax outputs.
  - Inspecting raw or processed data to debug issues in preprocessing or model training.

Example Notebooks:
  - predict_genre.ipynb: Loads an MP3 file, extracts features, runs the trained model, and visualizes genre prediction confidence.
  - analyze_dataset.ipynb: Displays class balance, feature histograms, and PCA/t-SNE visualizations of the feature space.

These notebooks are great for experimentation and visualization, especially when iterating on model performance or tuning preprocessing steps.

## Notes
Model Architecture: A hybrid CNN-LSTM model is used for genre classification. The CNN layers handle feature extraction from raw input data, and the LSTM layers capture temporal patterns in the audio features.

## Model Evaluation: The model's performance is evaluated using classification metrics like accuracy, precision, recall, and F1-score.