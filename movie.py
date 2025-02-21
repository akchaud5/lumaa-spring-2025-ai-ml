import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, movies_path='data/tmdb_5000_movies.csv'):
        """
        Initialize the MovieRecommender with the TMDB dataset.
        """
        # Define genre-specific terms with equal structure
        self.genre_terms = {
            'action': ['fight', 'explosion', 'chase', 'battle', 'combat', 'mission', 'hero', 'warrior', 'epic', 'spectacular'],
            'comedy': ['funny', 'humor', 'laugh', 'comedic', 'hilarious', 'witty', 'parody', 'satire', 'comedy', 'amusing'],
            'drama': ['emotional', 'relationship', 'conflict', 'intense', 'powerful', 'moving', 'dramatic', 'struggle', 'serious', 'dramatic'],
            'horror': ['scary', 'fear', 'terror', 'horror', 'frightening', 'supernatural', 'monster', 'creature', 'suspense', 'dark'],
            'romance': ['love', 'romance', 'relationship', 'romantic', 'passion', 'emotional', 'touching', 'heartfelt', 'tender', 'intimate'],
            'science fiction': ['space', 'future', 'technology', 'alien', 'planet', 'sci-fi', 'futuristic', 'advanced', 'scientific', 'innovation'],
            'thriller': ['suspense', 'tension', 'mystery', 'thriller', 'twist', 'suspenseful', 'intense', 'gripping', 'thrilling', 'exciting'],
            'adventure': ['journey', 'quest', 'expedition', 'discover', 'exploration', 'adventure', 'exciting', 'epic', 'discovery', 'voyage'],
            'fantasy': ['magic', 'magical', 'fantasy', 'mythical', 'supernatural', 'enchanted', 'mystical', 'fantastic', 'imaginative', 'legendary'],
            'animation': ['animated', 'cartoon', 'animation', 'colorful', 'imaginative', 'creative', 'visually', 'artistic', 'drawn', 'animated']
        }
        
        self.data = pd.read_csv(movies_path)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.preprocess_data()
        
    def _extract_json_field(self, json_str, field='name'):
        """Extract and clean fields from JSON string."""
        if pd.isna(json_str):
            return ''
        try:
            items = json.loads(json_str.replace("'", '"'))
            return ' '.join(item[field].lower() for item in items)
        except:
            return ''
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        text = str(text).lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        return ' '.join(text.split())
    
    def _boost_genre_terms(self, text):
        """Boost genre-related terms with equal weighting."""
        text_lower = text.lower()
        boosted_terms = [text_lower]
        
        # Find which genres are present
        for genre, terms in self.genre_terms.items():
            if genre in text_lower:
                # Add genre name for emphasis (equal boost for all genres)
                boosted_terms.extend([genre] * 2)
                # Add matching terms
                matching_terms = [term for term in terms if term in text_lower]
                boosted_terms.extend(matching_terms)
        
        return ' '.join(boosted_terms)
    
    def preprocess_data(self):
        """Preprocess the movie data."""
        # Fill NA values
        self.data['overview'] = self.data['overview'].fillna('')
        self.data['genres'] = self.data['genres'].fillna('[]')
        self.data['keywords'] = self.data['keywords'].fillna('[]')
        
        # Extract text from JSON fields
        self.data['genres_text'] = self.data['genres'].apply(self._extract_json_field)
        self.data['keywords_text'] = self.data['keywords'].apply(self._extract_json_field)
        
        # Clean and combine features
        features = []
        for _, row in self.data.iterrows():
            clean_overview = self._clean_text(row['overview'])
            clean_genres = self._boost_genre_terms(row['genres_text'])
            clean_keywords = self._clean_text(row['keywords_text']) * 2
            features.append(f"{clean_overview} {clean_genres} {clean_keywords}")
            
        self.data['combined_features'] = features
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['combined_features'])
    
    def get_recommendations(self, user_description, n_recommendations=3):
        """Get movie recommendations based on user description."""
        try:
            # Process user input
            processed_input = self._clean_text(user_description)
            processed_input = self._boost_genre_terms(processed_input)
            
            # Get similarity scores
            input_vector = self.vectorizer.transform([processed_input])
            similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix)[0]
            
            # Get indices of top similar movies
            similar_indices = similarity_scores.argsort()[::-1]
            
            # Filter and rank recommendations
            recommendations = []
            seen_titles = set()
            
            # Extract key terms from user query
            query_terms = set(processed_input.lower().split())
            
            for idx in similar_indices:
                if len(recommendations) >= n_recommendations:
                    break
                    
                movie = self.data.iloc[idx]
                
                # Skip duplicate titles
                if movie['title'] in seen_titles:
                    continue
                
                # Skip missing overviews
                if not movie['overview'] or len(movie['overview'].strip()) < 20:
                    continue
                
                # Get genres and overview terms
                genres = self._extract_json_field(movie['genres'])
                overview_terms = set(self._clean_text(movie['overview']).split())
                
                # Calculate content relevance
                content_matches = len(query_terms.intersection(overview_terms))
                
                # Adjust score based on content relevance
                base_score = similarity_scores[idx]
                final_score = base_score * (1 + content_matches * 0.1)
                
                # Add to recommendations if score is above threshold
                if final_score > 0.1:
                    recommendations.append({
                        'title': movie['title'],
                        'similarity_score': final_score,
                        'genres': genres,
                        'overview': movie['overview']
                    })
                    seen_titles.add(movie['title'])
            
            # Sort recommendations by similarity score in descending order
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('description', type=str, help='User preference description')
    parser.add_argument('--num_recommendations', type=int, default=3,
                      help='Number of recommendations to return')
    parser.add_argument('--data_path', type=str, default='data/tmdb_5000_movies.csv',
                      help='Path to movie dataset CSV file')
    
    args = parser.parse_args()
    
    try:
        # Initialize recommender
        recommender = MovieRecommender(args.data_path)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            args.description,
            args.num_recommendations
        )
        
        # Print recommendations
        print(f"\nTop {args.num_recommendations} recommendations based on: '{args.description}'\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Similarity Score: {rec['similarity_score']:.3f}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Overview: {rec['overview'][:200]}...\n")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())