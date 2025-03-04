# music_genre_model
A Python-based AI model that classifies music genres by analyzing audio features. It uses MusicBrainz API to pull songs across
a variety of genres. It stores genre, title, and artist information in a PostgreSQL database for both data collection purposes,
and to prevent duplicate songs from entering the database. Audio files are downloaded from Youtube.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Database](#database)
- [Scripts](#scripts)
- [Notes](#notes)

## Features
This model allows for the simple categorization of music. Feed the model a folder of songs, and it will determine what genre that song belongs to. I wrote this project to get my feet wet working with audio, and digital signal processing, as well as learning how to create and train my own AI model.

Core Functionalities:
  - Creating an AI model using Tensorflow.
  - Creation/collection of training data using MusicBrainz API, Youtube, Librosa, and sklearn
  - Storing song information, including genre, artist, and song title using PostgreSQL
  - Preventing duplicate songs from entering training using PostgreSQL
  - Training and saving the AI model in a .keras format for easy deployment using Tensorflow

Future direction of the project
  - To take this project further, I would likely add more genres
  - automatic expansion of genre headers is definitely possible, and likely necessary for a truly useful model
  - MusicBrainz API is hardly being used to its full capacity currently, it can produce very specific genre information.

## Installation
This project has a few prerequesites.
  - Python v3.11
  - PostgreSQL v16
  - FFmpeg (yt_dlp needs it)
  - pip

Installing FFmpeg:
For windows users, download from the FFmpeg website and add it to your system path.
Linux/MacOS users may use one of the following commands
- sudo apt install ffmpeg  # Debian/Ubuntu
- brew install ffmpeg  # macOS

Cloning Repository:
git clone https://github.com/aidantherrien/music_genre_model/tree/main
cd https://github.com/aidantherrien/music_genre_model/tree/main

Other Dependencies may be found in the requirements.txt file in the repository. Simply download, and run the following pip
command:
pip install -r requirements.txt

## Setup
Follow the followings steps before first using the program.
  - Start your PostgreSQL server (I used PGAdmin v4)
  - Add a user to your database, set a password (this will be how your program accesses your database)
  - Update login credentials in each script that uses it (Will be discussed further in scripts section)

## Database

Here is the layout of the songs table:
### `songs` Table
| Column       | Type        | Description                       |
|--------------|-------------|-----------------------------------|
| `song_id`    | SERIAL (PK) | Unique song identifier            |
| `title`      | TEXT        | Song title                        |
| `artist`     | TEXT        | Artist name                       |
| `genre`      | TEXT        | Genre label from MusicBrainzAPI   |
| `trained`    | BOOLEAN     | True/False has the model used it  |

--- 

Each pass by MusicBrainzAPI will add 800 songs to the SQL table, 100 songs per genre. This is also where duplicate songs
are prevented from entering the table using "ON CONFLICT DO NOTHING".

Here is the create table query.
CREATE TABLE songs (
    song_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    genre TEXT,
    trained BOOLEAN DEFAULT FALSE
); 

## Scripts
create_new_model.py
  - Creates a new genre classification model using Tensorflow
  - Input your own desired filename at the top
  - Adjust layers of each model here
  - Models are save to the "Models" directory

train_and_save_model.py
  - Trains your model using the preprocessed data in the relevant .npz file.
  - You must input the path to the model you wish to train
  - You must input the path to the .npz data you wish to use in training
  - Script will only save the new model if the accuracy has improved over the previous version.

musicbrainz_collection.py
  - Pulls songs and genre information from the MusicBrainzAPI database.
  - Data is placed into the songs table in the SQL database
  - You must input your own SQL login information at the top
  - You may input your own MusicBrainzAPI contact information in the set_useragent() function at line 15

download_from_youtube.py
  - Pulls song names from the PostgreSQL table, searches for them on Youtube, and download the audio file
  - Audio files are places in genreMode/unprocessed/?
  - ? represents a genre header, mp3 files are sorted while downloading into labeled folders
  - Note that this function may take up to 7GB of disk space.
  - This script takes a while to run, close to an hour and a half on my machine

extract_audio.py
  - This is where you get the .npz used by train_and_save_model.py
  - You must go down to the main block at the bottom, and change data_path to the currentData/unprocessed folder.
  - You may also change where the .npz folder winds up, I placed it in currentData/processed.

## Notes
I have a bunch of old scripts (nicely placed in the outdated scripts directory). Those were used while I was initially
tooling around and testing some of this code. Feel free to use any of it for your own troubleshooting.
 

 





