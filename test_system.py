"""
Test script to verify all components of the Anime Recommendation System
"""

import sys
import os


def test_imports():
    """Test that all required libraries can be imported"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        print("✓ All required libraries are installed")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_dataset():
    """Test that the dataset can be loaded"""
    print("\nTesting dataset loading...")
    try:
        import pandas as pd
        df = pd.read_csv('anime_recommendation_dataset.csv')
        expected_columns = ['title', 'synopsis', 'genres', 'episodes', 'score', 'characters']
        
        if list(df.columns) != expected_columns:
            print(f"✗ Unexpected columns: {list(df.columns)}")
            return False
        
        if len(df) == 0:
            print("✗ Dataset is empty")
            return False
            
        print(f"✓ Dataset loaded successfully ({len(df)} anime)")
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False


def test_recommender_init():
    """Test that the recommender can be initialized"""
    print("\nTesting recommender initialization...")
    try:
        from anime_recommender import AnimeRecommender
        recommender = AnimeRecommender()
        print("✓ Recommender initialized successfully")
        return True, recommender
    except Exception as e:
        print(f"✗ Recommender initialization error: {e}")
        return False, None


def test_recommendations(recommender):
    """Test the recommendation function"""
    print("\nTesting get_recommendations...")
    try:
        recs = recommender.get_recommendations('Cowboy Bebop', n_recommendations=5)
        if len(recs) == 0:
            print("✗ No recommendations returned")
            return False
        print(f"✓ Recommendations generated ({len(recs)} results)")
        return True
    except Exception as e:
        print(f"✗ Recommendations error: {e}")
        return False


def test_genre_recommendations(recommender):
    """Test genre-based recommendations"""
    print("\nTesting recommend_by_genre...")
    try:
        recs = recommender.recommend_by_genre('Action', n_recommendations=5, min_score=70)
        if len(recs) == 0:
            print("⚠ No genre recommendations found (might be due to filters)")
        else:
            print(f"✓ Genre recommendations generated ({len(recs)} results)")
        return True
    except Exception as e:
        print(f"✗ Genre recommendations error: {e}")
        return False


def test_search(recommender):
    """Test search functionality"""
    print("\nTesting search_anime...")
    try:
        results = recommender.search_anime('Hunter')
        # Search might return empty results, which is valid
        print(f"✓ Search completed ({len(results)} results)")
        return True
    except Exception as e:
        print(f"✗ Search error: {e}")
        return False


def test_top_rated(recommender):
    """Test top rated functionality"""
    print("\nTesting get_top_rated...")
    try:
        top = recommender.get_top_rated(n=5)
        if len(top) == 0:
            print("✗ No top rated anime found")
            return False
        print(f"✓ Top rated anime retrieved ({len(top)} results)")
        return True
    except Exception as e:
        print(f"✗ Top rated error: {e}")
        return False


def test_random_recommendations(recommender):
    """Test random recommendations"""
    print("\nTesting get_random_recommendations...")
    try:
        random_recs = recommender.get_random_recommendations(n=3, min_score=70)
        if len(random_recs) == 0:
            print("⚠ No random recommendations found (might be due to filters)")
        else:
            print(f"✓ Random recommendations generated ({len(random_recs)} results)")
        return True
    except Exception as e:
        print(f"✗ Random recommendations error: {e}")
        return False


def test_visualizations():
    """Test that visualization script can be imported"""
    print("\nTesting visualization script...")
    try:
        import visualize_data
        print("✓ Visualization script can be imported")
        return True
    except Exception as e:
        print(f"✗ Visualization script error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("ANIME RECOMMENDATION SYSTEM - TEST SUITE")
    print("="*80)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Imports
    tests_total += 1
    if test_imports():
        tests_passed += 1
    
    # Test 2: Dataset
    tests_total += 1
    if test_dataset():
        tests_passed += 1
    
    # Test 3: Recommender initialization
    tests_total += 1
    success, recommender = test_recommender_init()
    if success:
        tests_passed += 1
    
    # Only run remaining tests if recommender initialized
    if recommender:
        # Test 4: Basic recommendations
        tests_total += 1
        if test_recommendations(recommender):
            tests_passed += 1
        
        # Test 5: Genre recommendations
        tests_total += 1
        if test_genre_recommendations(recommender):
            tests_passed += 1
        
        # Test 6: Search
        tests_total += 1
        if test_search(recommender):
            tests_passed += 1
        
        # Test 7: Top rated
        tests_total += 1
        if test_top_rated(recommender):
            tests_passed += 1
        
        # Test 8: Random recommendations
        tests_total += 1
        if test_random_recommendations(recommender):
            tests_passed += 1
    
    # Test 9: Visualizations
    tests_total += 1
    if test_visualizations():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} tests passed")
    print("="*80)
    
    if tests_passed == tests_total:
        print("✓ All tests passed! The recommendation system is working correctly.")
        return 0
    else:
        print(f"✗ {tests_total - tests_passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
