# Genre list (in order corresponding to neurons in the model)
GENRES = [
    "classic_rock",
    "alternative_rock",
    "alternative",
    "pop_punk",
    "punk",
    "soul",
    "motown",
    "funk",
    "disco",
    "hip-hop",
    "rap",
    "folk",
    "country",
    "pop_country",
    "fusion",
    "jazz",
    "classical",
    "blues",
    "metal",
    "heavy_metal",
    "rock",
    "pop",
    "electronic"
]

# Mapping from genre to class index
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

# Mapping from class index back to genre
INDEX_TO_GENRE = {i: genre for i, genre in enumerate(GENRES)}