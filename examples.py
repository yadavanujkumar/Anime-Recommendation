"""
Quick Start Examples for Anime Recommendation System
Run this file to see the recommendation system in action!
"""

from anime_recommender import AnimeRecommender


def example_1_basic_recommendations():
    """Example 1: Get recommendations for a specific anime"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Recommendations")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    # Get recommendations similar to "Cowboy Bebop"
    anime_title = "Cowboy Bebop"
    print(f"\nFinding anime similar to '{anime_title}'...")
    recommendations = recommender.get_recommendations(anime_title, n_recommendations=5)
    print("\nTop 5 Recommendations:")
    print(recommendations.to_string(index=False))


def example_2_genre_recommendations():
    """Example 2: Get recommendations by genre"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Genre-Based Recommendations")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    # Find top Action anime
    print("\nTop 5 Action Anime (score >= 75):")
    action_anime = recommender.recommend_by_genre('Action', n_recommendations=5, min_score=75)
    print(action_anime.to_string(index=False))
    
    # Find top Romance anime
    print("\n\nTop 5 Romance Anime (score >= 70):")
    romance_anime = recommender.recommend_by_genre('Romance', n_recommendations=5, min_score=70)
    print(romance_anime.to_string(index=False))


def example_3_search():
    """Example 3: Search for anime"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Search Functionality")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    # Search for anime with "Dragon" in the title
    search_term = "Dragon"
    print(f"\nSearching for anime with '{search_term}' in title...")
    results = recommender.search_anime(search_term)
    if len(results) > 0:
        print(results.to_string(index=False))
    else:
        print(f"No results found for '{search_term}'")


def example_4_top_rated():
    """Example 4: Get top rated anime"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Top Rated Anime")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    print("\nTop 10 Highest Rated Anime:")
    top_anime = recommender.get_top_rated(n=10)
    print(top_anime.to_string(index=False))


def example_5_discover():
    """Example 5: Discover random high-quality anime"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Discover Random High-Quality Anime")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    print("\nRandom recommendations (score >= 75):")
    random_anime = recommender.get_random_recommendations(n=5, min_score=75)
    print(random_anime.to_string(index=False))


def example_6_custom_workflow():
    """Example 6: Custom workflow - Find similar anime to your favorites"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Workflow")
    print("="*80)
    
    recommender = AnimeRecommender()
    
    # Step 1: Search for an anime
    print("\nStep 1: Search for 'Naruto'")
    search_results = recommender.search_anime('Naruto')
    print(search_results.to_string(index=False))
    
    # Step 2: Get recommendations
    if len(search_results) > 0:
        anime_title = search_results.iloc[0]['title']
        print(f"\n\nStep 2: Get recommendations similar to '{anime_title}'")
        recommendations = recommender.get_recommendations(anime_title, n_recommendations=5)
        print(recommendations.to_string(index=False))


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ANIME RECOMMENDATION SYSTEM - EXAMPLES")
    print("="*80)
    
    try:
        example_1_basic_recommendations()
        example_2_genre_recommendations()
        example_3_search()
        example_4_top_rated()
        example_5_discover()
        example_6_custom_workflow()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nTry modifying the examples above to explore different anime!")
        print("Check out 'anime_recommender.py' for more functionality.")
        print("\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the dataset file 'anime_recommendation_dataset.csv' exists.")


if __name__ == "__main__":
    main()
