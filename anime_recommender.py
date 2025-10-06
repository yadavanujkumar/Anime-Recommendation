"""
Anime Recommendation System
A content-based recommendation engine for anime using NLP and machine learning
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


class AnimeRecommender:
    """
    A content-based anime recommendation system
    """
    
    def __init__(self, data_path='anime_recommendation_dataset.csv'):
        """
        Initialize the recommender system
        
        Parameters:
        -----------
        data_path : str
            Path to the anime dataset CSV file
        """
        self.df = pd.read_csv(data_path)
        self.df_processed = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.tfidf_vectorizer = None
        
        print(f"Loaded {len(self.df)} anime entries")
        self._preprocess_data()
        self._build_recommendation_engine()
    
    def _clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', str(text))
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def _preprocess_data(self):
        """Preprocess the anime data"""
        print("Preprocessing data...")
        self.df_processed = self.df.copy()
        
        # Clean text fields
        self.df_processed['cleaned_synopsis'] = self.df_processed['synopsis'].apply(self._clean_text)
        self.df_processed['cleaned_genres'] = self.df_processed['genres'].fillna('').str.lower()
        self.df_processed['cleaned_characters'] = self.df_processed['characters'].fillna('').apply(
            lambda x: ' '.join([c.strip() for c in str(x).split(',')[:10]])
        ).apply(self._clean_text)
        
        # Create combined features (give more weight to genres)
        self.df_processed['combined_features'] = (
            self.df_processed['cleaned_synopsis'] + ' ' + 
            self.df_processed['cleaned_genres'].str.replace(',', ' ').str.repeat(3) + ' ' +
            self.df_processed['cleaned_characters']
        )
        
        self.df_processed['combined_features'] = self.df_processed['combined_features'].fillna('')
        print("Data preprocessing completed!")
    
    def _build_recommendation_engine(self):
        """Build the TF-IDF matrix and compute similarity"""
        print("Building recommendation engine...")
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        # Fit and transform
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df_processed['combined_features'])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create indices for quick lookup
        self.indices = pd.Series(self.df_processed.index, index=self.df_processed['title']).drop_duplicates()
        
        print("Recommendation engine built successfully!")
    
    def get_recommendations(self, title, n_recommendations=10, score_weight=0.3):
        """
        Get anime recommendations based on content similarity
        
        Parameters:
        -----------
        title : str
            The title of the anime
        n_recommendations : int
            Number of recommendations to return
        score_weight : float
            Weight for incorporating anime score (0 to 1)
        
        Returns:
        --------
        DataFrame with recommended anime
        """
        try:
            # Get the index of the anime
            idx = self.indices[title]
            
            # Get pairwise similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Incorporate score rating
            if score_weight > 0:
                max_score = self.df_processed['score'].max()
                min_score = self.df_processed['score'].min()
                normalized_scores = (self.df_processed['score'] - min_score) / (max_score - min_score)
                
                sim_scores = [(i, score * (1 - score_weight) + normalized_scores.iloc[i] * score_weight) 
                             for i, score in sim_scores]
            
            # Sort by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n similar anime (excluding the input anime itself)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            # Get anime indices
            anime_indices = [i[0] for i in sim_scores]
            similarity_scores = [i[1] for i in sim_scores]
            
            # Return the top n most similar anime
            recommendations = self.df.iloc[anime_indices][['title', 'genres', 'score', 'episodes']].copy()
            recommendations['similarity_score'] = similarity_scores
            
            return recommendations
            
        except KeyError:
            print(f"Anime '{title}' not found in the database.")
            print("Did you mean one of these?")
            similar = self.search_anime(title, top_n=5)
            return similar
    
    def recommend_by_genre(self, genre, n_recommendations=10, min_score=60):
        """
        Recommend anime by genre
        
        Parameters:
        -----------
        genre : str
            Genre to filter by
        n_recommendations : int
            Number of recommendations
        min_score : float
            Minimum score threshold
        
        Returns:
        --------
        DataFrame with recommended anime
        """
        filtered_df = self.df[
            (self.df['genres'].str.contains(genre, case=False, na=False)) & 
            (self.df['score'] >= min_score)
        ].sort_values('score', ascending=False)
        
        return filtered_df[['title', 'genres', 'score', 'episodes']].head(n_recommendations)
    
    def search_anime(self, query, top_n=10):
        """
        Search for anime by partial title match
        
        Parameters:
        -----------
        query : str
            Search query
        top_n : int
            Number of results to return
        
        Returns:
        --------
        DataFrame with matching anime
        """
        mask = self.df['title'].str.contains(query, case=False, na=False)
        results = self.df[mask][['title', 'genres', 'score', 'episodes']].sort_values('score', ascending=False)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        return results.head(top_n)
    
    def get_top_rated(self, n=10, min_episodes=1):
        """
        Get top rated anime
        
        Parameters:
        -----------
        n : int
            Number of anime to return
        min_episodes : int
            Minimum episode count filter
        
        Returns:
        --------
        DataFrame with top rated anime
        """
        filtered = self.df[self.df['episodes'] >= min_episodes]
        return filtered.nlargest(n, 'score')[['title', 'genres', 'score', 'episodes']]
    
    def get_random_recommendations(self, n=5, min_score=70):
        """
        Get random high-quality anime recommendations
        
        Parameters:
        -----------
        n : int
            Number of random recommendations
        min_score : float
            Minimum score threshold
        
        Returns:
        --------
        DataFrame with random anime
        """
        filtered = self.df[self.df['score'] >= min_score]
        if len(filtered) < n:
            n = len(filtered)
        return filtered.sample(n=n)[['title', 'genres', 'score', 'episodes']]
    
    def get_anime_info(self, title):
        """
        Get detailed information about a specific anime
        
        Parameters:
        -----------
        title : str
            Anime title
        
        Returns:
        --------
        Series with anime information
        """
        mask = self.df['title'].str.contains(f'^{title}$', case=False, na=False, regex=True)
        results = self.df[mask]
        
        if len(results) == 0:
            print(f"Anime '{title}' not found. Did you mean:")
            similar = self.search_anime(title, top_n=5)
            return similar
        
        return results.iloc[0]


def main():
    """Example usage of the AnimeRecommender"""
    
    # Initialize the recommender
    recommender = AnimeRecommender()
    
    print("\n" + "="*80)
    print("ANIME RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Example 1: Get recommendations for Cowboy Bebop
    print("\n1. Recommendations similar to 'Cowboy Bebop':")
    print("-" * 80)
    recs = recommender.get_recommendations('Cowboy Bebop', n_recommendations=5)
    print(recs.to_string(index=False))
    
    # Example 2: Get top Action anime
    print("\n2. Top Action Anime (score >= 70):")
    print("-" * 80)
    action_recs = recommender.recommend_by_genre('Action', n_recommendations=5, min_score=70)
    print(action_recs.to_string(index=False))
    
    # Example 3: Search for anime
    print("\n3. Search results for 'Hunter':")
    print("-" * 80)
    search_results = recommender.search_anime('Hunter', top_n=5)
    print(search_results.to_string(index=False))
    
    # Example 4: Get top rated anime
    print("\n4. Top 5 Rated Anime:")
    print("-" * 80)
    top_rated = recommender.get_top_rated(n=5)
    print(top_rated.to_string(index=False))
    
    # Example 5: Random recommendations
    print("\n5. Random High-Quality Recommendations:")
    print("-" * 80)
    random_recs = recommender.get_random_recommendations(n=3, min_score=75)
    print(random_recs.to_string(index=False))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
