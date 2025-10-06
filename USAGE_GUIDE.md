# Anime Recommendation System - Usage Guide

## Overview
This guide provides detailed instructions on how to use the Anime Recommendation System effectively.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yadavanujkumar/Anime-Recommendation.git
cd Anime-Recommendation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage Options

### Option 1: Jupyter Notebook (Recommended for Learning)

The Jupyter notebook provides an interactive environment with visualizations and step-by-step explanations.

```bash
jupyter notebook anime_recommendation_system.ipynb
```

**Features in the notebook:**
- Complete EDA with visualizations
- NLP preprocessing demonstrations
- Interactive recommendation examples
- Detailed explanations of each step

### Option 2: Python Script (For Integration)

Use the `anime_recommender.py` module in your own Python scripts.

```python
from anime_recommender import AnimeRecommender

# Initialize the recommender
recommender = AnimeRecommender()

# Get recommendations
recommendations = recommender.get_recommendations('Cowboy Bebop', n_recommendations=10)
print(recommendations)
```

### Option 3: Example Scripts

Run pre-built examples:
```bash
python examples.py
```

### Option 4: Visualizations

Generate all visualizations:
```bash
python visualize_data.py
```

This creates:
- `score_distribution.png` - Score distribution analysis
- `episode_distribution.png` - Episode count patterns
- `genre_distribution.png` - Most popular genres
- `genre_avg_scores.png` - Best-performing genres
- `word_frequency.png` - Common words in synopsis
- `score_vs_episodes.png` - Correlation analysis
- `summary_statistics.txt` - Text summary of dataset

## API Reference

### AnimeRecommender Class

#### `__init__(data_path='anime_recommendation_dataset.csv')`
Initialize the recommender with a dataset.

**Parameters:**
- `data_path` (str): Path to the CSV dataset

#### `get_recommendations(title, n_recommendations=10, score_weight=0.3)`
Get anime recommendations based on similarity.

**Parameters:**
- `title` (str): Title of the anime to find similar recommendations for
- `n_recommendations` (int): Number of recommendations to return
- `score_weight` (float): Weight for anime score (0-1), higher means more weight on ratings

**Returns:**
- DataFrame with columns: title, genres, score, episodes, similarity_score

**Example:**
```python
recs = recommender.get_recommendations('Cowboy Bebop', n_recommendations=5, score_weight=0.3)
```

#### `recommend_by_genre(genre, n_recommendations=10, min_score=60)`
Get top anime from a specific genre.

**Parameters:**
- `genre` (str): Genre to filter by (e.g., 'Action', 'Comedy', 'Romance')
- `n_recommendations` (int): Number of results to return
- `min_score` (float): Minimum score threshold

**Returns:**
- DataFrame with anime matching criteria

**Example:**
```python
action_anime = recommender.recommend_by_genre('Action', n_recommendations=10, min_score=75)
```

#### `search_anime(query, top_n=10)`
Search for anime by partial title match.

**Parameters:**
- `query` (str): Search term
- `top_n` (int): Maximum number of results

**Returns:**
- DataFrame with matching anime, sorted by score

**Example:**
```python
results = recommender.search_anime('Hunter')
```

#### `get_top_rated(n=10, min_episodes=1)`
Get the highest-rated anime.

**Parameters:**
- `n` (int): Number of anime to return
- `min_episodes` (int): Minimum episode count filter

**Returns:**
- DataFrame with top-rated anime

**Example:**
```python
top = recommender.get_top_rated(n=20, min_episodes=10)
```

#### `get_random_recommendations(n=5, min_score=70)`
Get random high-quality anime for discovery.

**Parameters:**
- `n` (int): Number of random anime to return
- `min_score` (float): Minimum score threshold

**Returns:**
- DataFrame with random high-quality anime

**Example:**
```python
discover = recommender.get_random_recommendations(n=5, min_score=75)
```

#### `get_anime_info(title)`
Get detailed information about a specific anime.

**Parameters:**
- `title` (str): Anime title (exact match)

**Returns:**
- Series with anime information

**Example:**
```python
info = recommender.get_anime_info('Cowboy Bebop')
print(info)
```

## Common Use Cases

### Use Case 1: Find Similar Anime
If you liked a specific anime and want to find similar ones:

```python
from anime_recommender import AnimeRecommender

recommender = AnimeRecommender()
similar = recommender.get_recommendations('Your Favorite Anime', n_recommendations=10)
print(similar)
```

### Use Case 2: Discover by Genre
Explore the best anime in a specific genre:

```python
# Find top comedy anime
comedy = recommender.recommend_by_genre('Comedy', n_recommendations=15, min_score=70)

# Find highly-rated action anime
action = recommender.recommend_by_genre('Action', n_recommendations=10, min_score=80)
```

### Use Case 3: Search and Explore
Search for anime and get recommendations:

```python
# Search for an anime
search_results = recommender.search_anime('Naruto')
print("Search Results:")
print(search_results)

# Get recommendations based on first result
if len(search_results) > 0:
    anime_title = search_results.iloc[0]['title']
    recommendations = recommender.get_recommendations(anime_title, n_recommendations=5)
    print(f"\nSimilar to {anime_title}:")
    print(recommendations)
```

### Use Case 4: Daily Recommendations
Get fresh recommendations each day:

```python
# Get random high-quality anime
daily_picks = recommender.get_random_recommendations(n=3, min_score=75)
print("Today's Recommendations:")
print(daily_picks)
```

### Use Case 5: Build a Watchlist
Create a personalized watchlist:

```python
watchlist = []

# Add top-rated anime
top_anime = recommender.get_top_rated(n=5, min_episodes=12)
watchlist.extend(top_anime['title'].tolist())

# Add recommendations based on favorites
favorites = ['Cowboy Bebop', 'TRIGUN']
for favorite in favorites:
    recs = recommender.get_recommendations(favorite, n_recommendations=3)
    watchlist.extend(recs['title'].tolist())

# Remove duplicates
watchlist = list(set(watchlist))
print(f"Your personalized watchlist ({len(watchlist)} anime):")
for anime in watchlist:
    print(f"  - {anime}")
```

## Tips for Best Results

1. **Score Weight Parameter**: 
   - Use `score_weight=0` for pure content-based similarity
   - Use `score_weight=0.3-0.5` for a balance (recommended)
   - Use `score_weight=0.7-1.0` to prioritize highly-rated anime

2. **Genre Filtering**:
   - Genres are case-insensitive
   - Use specific genres for better results (e.g., 'Sci-Fi' instead of 'Science')
   - Combine with `min_score` to filter low-rated content

3. **Search Tips**:
   - Search is case-insensitive and uses partial matching
   - If exact title doesn't work, try keywords or partial names
   - Results are sorted by score by default

4. **Exploration**:
   - Use `get_random_recommendations()` to discover new anime
   - Adjust `min_score` threshold based on your preferences
   - Try different genres to expand your anime horizons

## Troubleshooting

### Issue: "Anime not found in database"
**Solution:** 
- Use the search function to find the correct title
- Check for spelling errors
- The dataset contains specific anime titles - use exact matches

### Issue: "No recommendations returned"
**Solution:**
- Check if the anime exists in the dataset using search
- Verify the dataset file is in the correct location
- Ensure all dependencies are installed

### Issue: "Import errors"
**Solution:**
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Verify scikit-learn is properly installed

## Performance Considerations

- **Initial Load Time**: First initialization takes ~2-5 seconds for preprocessing
- **Recommendation Speed**: Individual recommendations are near-instantaneous
- **Memory Usage**: ~50-100MB for the full dataset and similarity matrix
- **Scalability**: Current implementation works well for datasets up to 10,000 anime

## Dataset Information

The dataset contains:
- 200 anime entries
- Features: title, synopsis, genres, episodes, score, characters
- Score range: 0-100
- Multiple genres per anime
- Character lists (up to 10 main characters used)

## Future Enhancements

Planned improvements:
- [ ] Add user rating integration
- [ ] Implement collaborative filtering
- [ ] Create web interface
- [ ] Add more filtering options (year, studio, etc.)
- [ ] Expand dataset with more anime
- [ ] Add anime poster images
- [ ] Implement deep learning models

## Getting Help

- Check the examples in `examples.py`
- Review the Jupyter notebook for detailed explanations
- Read the code documentation in `anime_recommender.py`
- Check the README for project overview

## Contributing

We welcome contributions! Areas for improvement:
- Dataset expansion
- Algorithm enhancements
- Documentation improvements
- Bug fixes
- New features

---

Last updated: 2024
