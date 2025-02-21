# Movie Recommendation System
A content-based movie recommender using TF-IDF and cosine similarity.

# Dataset

Uses TMDB 5000 Movies Dataset (tmdb_5000_movies.csv)
Store in data/ directory
Contains movie titles, overviews, genres, and metadata

Setup Requirements
- Python 3.13.1
- Virtual environment

# Create and activate environment
python3 -m venv venv

source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Usage

python3 movie.py "I love science fiction movies with amazing visual effects"

# Optional arguments
--num_recommendations N  # Number of recommendations (default: 3)

--data_path PATH        # Custom dataset path

# Sample Output

Top 3 recommendations based on: 'I love science fiction movies with amazing visual effects'

1. Space Pirate Captain Harlock (0.594)
   Genres: animation, science fiction
   Overview: Space Pirate Captain Harlock and his fearless crew face off...

2. Lost in Space (0.498)
   Genres: adventure, family, science fiction
   Overview: The prospects for continuing life on Earth in 2058 are grim...

3. Invaders from Mars (0.453)
   Genres: science fiction
   Overview: In this remake of the classic 50s SF tale...
Dependencies

pandas
numpy
scikit-learn
