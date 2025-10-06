# Anime Recommendation System 🎌

A comprehensive anime recommendation system built using Natural Language Processing (NLP) and Machine Learning techniques. This project includes exploratory data analysis (EDA), text preprocessing, and a content-based recommendation engine.

## 📊 Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of anime dataset with visualizations
- **Natural Language Processing**: Text preprocessing, cleaning, and TF-IDF vectorization
- **Content-Based Recommendation Engine**: Recommends anime based on similarity using cosine similarity
- **Multiple Recommendation Methods**:
  - Similar anime recommendations based on content
  - Genre-based recommendations
  - Search functionality
  - Top-rated anime
  - Random high-quality recommendations

## 📁 Dataset

The dataset (`anime_recommendation_dataset.csv`) contains **557 anime entries** with the following features:
- **title**: Name of the anime
- **synopsis**: Plot description
- **genres**: Comma-separated list of genres
- **episodes**: Number of episodes
- **score**: Rating score (0-100)
- **characters**: Main characters

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Anime-Recommendation.git
cd Anime-Recommendation
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Using Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook anime_recommendation_system.ipynb
```

The notebook includes:
- Complete EDA with visualizations
- NLP preprocessing steps
- Recommendation engine implementation
- Interactive examples

#### Option 2: Using Python Script

```python
from anime_recommender import AnimeRecommender

# Initialize the recommender
recommender = AnimeRecommender()

# Get recommendations for a specific anime
recommendations = recommender.get_recommendations('Cowboy Bebop', n_recommendations=10)
print(recommendations)

# Get recommendations by genre
action_anime = recommender.recommend_by_genre('Action', n_recommendations=10, min_score=70)
print(action_anime)

# Search for anime
results = recommender.search_anime('Hunter')
print(results)

# Get top rated anime
top_rated = recommender.get_top_rated(n=10)
print(top_rated)

# Get random high-quality recommendations
random_recs = recommender.get_random_recommendations(n=5, min_score=75)
print(random_recs)
```

Or simply run the example:
```bash
python anime_recommender.py
```

## 📈 EDA Insights

### Key Findings:

1. **Score Distribution**: Most anime have scores between 60-80
2. **Popular Genres**: Action, Comedy, Drama, and Adventure are the most common
3. **Episode Counts**: Many anime have 12-26 episodes (standard seasons)
4. **Genre Performance**: Different genres have varying average scores

### Visualizations Include:
- Score distribution (histogram and box plot)
- Episode count distribution
- Top genres analysis
- Genre vs. average score
- Word frequency analysis from synopsis

## 🤖 How the Recommendation Engine Works

The recommendation system uses **Content-Based Filtering**:

1. **Text Preprocessing**:
   - Removes HTML tags and special characters
   - Converts text to lowercase
   - Cleans and normalizes text

2. **Feature Engineering**:
   - Combines synopsis, genres (with higher weight), and characters
   - Creates a unified feature vector for each anime

3. **TF-IDF Vectorization**:
   - Converts text to numerical features
   - Uses both unigrams and bigrams
   - Captures importance of terms across documents

4. **Cosine Similarity**:
   - Computes similarity between all anime pairs
   - Recommends anime with highest similarity scores
   - Optional: Incorporates anime ratings for better recommendations

## 📊 Example Recommendations

### For "Cowboy Bebop":
Similar anime include sci-fi/action titles with space themes and bounty hunter narratives.

### Top Action Anime:
Filtered by genre and rating to show highest-quality action anime.

### Search Functionality:
Find anime by partial title match, sorted by score.

## 🛠️ Technical Stack

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **matplotlib & seaborn**: Data visualization
- **Jupyter Notebook**: Interactive analysis

## 📝 Project Structure

```
Anime-Recommendation/
├── anime_recommendation_dataset.csv    # Dataset
├── anime_recommendation_system.ipynb   # Jupyter notebook with full analysis
├── anime_recommender.py                # Python script version
├── requirements.txt                    # Dependencies
├── README.md                          # Documentation
└── LICENSE                            # MIT License
```

## 🎯 Future Enhancements

Potential improvements for the recommendation system:
- Collaborative filtering based on user ratings
- Hybrid recommendation approach
- Deep learning models (neural collaborative filtering)
- User interface (web app or dashboard)
- API endpoint for recommendations
- Integration with anime databases (MyAnimeList, AniList)
- Sentiment analysis on reviews

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 👤 Author

**Anuj Kumar Yadav**

## 🙏 Acknowledgments

- Dataset sourced from anime databases
- Inspired by content-based recommendation systems
- Built with open-source machine learning libraries

---

**Note**: This is a content-based recommendation system. For best results, the dataset should be expanded and potentially combined with user rating data for collaborative filtering approaches.