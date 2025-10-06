"""
Anime Data Visualization Script
Generates comprehensive visualizations for the anime dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')


def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def plot_score_distribution(df):
    """Plot score distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df['score'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Anime Scores', fontsize=14, fontweight='bold')
    axes[0].axvline(df['score'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {df["score"].mean():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df['score'].dropna(), vert=True)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: score_distribution.png")
    plt.close()


def plot_episode_distribution(df):
    """Plot episode distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram (limited to reasonable range)
    episodes_filtered = df[df['episodes'] <= 100]['episodes'].dropna()
    axes[0].hist(episodes_filtered, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0].set_xlabel('Number of Episodes (≤100)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Episode Count', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Top 10 most common episode counts
    episode_counts = df['episodes'].value_counts().head(10)
    axes[1].barh(episode_counts.index.astype(str), episode_counts.values, color='lightcoral')
    axes[1].set_xlabel('Count', fontsize=12)
    axes[1].set_ylabel('Number of Episodes', fontsize=12)
    axes[1].set_title('Top 10 Most Common Episode Counts', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('episode_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: episode_distribution.png")
    plt.close()


def plot_genre_analysis(df):
    """Plot genre analysis"""
    # Extract all genres
    all_genres = []
    for genres in df['genres'].dropna():
        genre_list = [g.strip() for g in genres.split(',')]
        all_genres.extend(genre_list)
    
    genre_counts = Counter(all_genres)
    top_genres = pd.DataFrame(genre_counts.most_common(15), columns=['Genre', 'Count'])
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    plt.barh(top_genres['Genre'], top_genres['Count'], color=colors, edgecolor='black')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title('Top 15 Most Common Anime Genres', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('genre_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: genre_distribution.png")
    plt.close()
    
    return genre_counts


def plot_genre_scores(df):
    """Plot average scores by genre"""
    genre_scores = {}
    for idx, row in df.iterrows():
        if pd.notna(row['genres']) and pd.notna(row['score']):
            genres = [g.strip() for g in row['genres'].split(',')]
            for genre in genres:
                if genre not in genre_scores:
                    genre_scores[genre] = []
                genre_scores[genre].append(row['score'])
    
    # Calculate average score for each genre (minimum 5 anime)
    genre_avg_scores = {genre: np.mean(scores) for genre, scores in genre_scores.items() 
                       if len(scores) >= 5}
    genre_avg_df = pd.DataFrame(list(genre_avg_scores.items()), columns=['Genre', 'Avg_Score'])
    genre_avg_df = genre_avg_df.sort_values('Avg_Score', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(genre_avg_df)))
    plt.barh(genre_avg_df['Genre'], genre_avg_df['Avg_Score'], color=colors, edgecolor='black')
    plt.xlabel('Average Score', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title('Top 15 Genres by Average Score (min 5 anime)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('genre_avg_scores.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: genre_avg_scores.png")
    plt.close()


def plot_word_frequency(df):
    """Plot word frequency from synopsis"""
    # Get all words from synopsis
    all_words = []
    for synopsis in df['synopsis'].dropna():
        cleaned = clean_text(synopsis)
        all_words.extend(cleaned.split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                  'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                  'should', 'could', 'may', 'might', 'can', 'his', 'her', 'their', 'our', 
                  'your', 'its', 'this', 'that', 'these', 'those', 'he', 'she', 'it', 
                  'they', 'them', 'who', 'which', 'what', 'when', 'where', 'why', 'how'}
    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 3]
    
    word_freq = Counter(filtered_words)
    top_words = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
    
    plt.figure(figsize=(14, 8))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(top_words)))
    plt.barh(top_words['Word'], top_words['Frequency'], color=colors, edgecolor='black')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    plt.title('Top 20 Most Common Words in Anime Synopsis', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('word_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: word_frequency.png")
    plt.close()


def plot_score_vs_episodes(df):
    """Plot relationship between score and episodes"""
    # Filter to reasonable episode range
    filtered_df = df[(df['episodes'] <= 200) & (df['episodes'] > 0)].dropna(subset=['score', 'episodes'])
    
    plt.figure(figsize=(12, 6))
    plt.scatter(filtered_df['episodes'], filtered_df['score'], alpha=0.5, s=50, c='steelblue', edgecolors='black')
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Anime Score vs Number of Episodes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(filtered_df['episodes'], filtered_df['score'], 1)
    p = np.poly1d(z)
    plt.plot(filtered_df['episodes'].sort_values(), 
             p(filtered_df['episodes'].sort_values()), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('score_vs_episodes.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: score_vs_episodes.png")
    plt.close()


def generate_summary_stats(df):
    """Generate and save summary statistics"""
    with open('summary_statistics.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANIME DATASET SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Anime Entries: {len(df)}\n\n")
        
        f.write("SCORE STATISTICS:\n")
        f.write(f"  Mean Score: {df['score'].mean():.2f}\n")
        f.write(f"  Median Score: {df['score'].median():.2f}\n")
        f.write(f"  Min Score: {df['score'].min():.2f}\n")
        f.write(f"  Max Score: {df['score'].max():.2f}\n")
        f.write(f"  Std Dev: {df['score'].std():.2f}\n\n")
        
        f.write("EPISODE STATISTICS:\n")
        f.write(f"  Mean Episodes: {df['episodes'].mean():.2f}\n")
        f.write(f"  Median Episodes: {df['episodes'].median():.2f}\n")
        f.write(f"  Min Episodes: {df['episodes'].min():.0f}\n")
        f.write(f"  Max Episodes: {df['episodes'].max():.0f}\n\n")
        
        f.write("TOP 10 HIGHEST RATED ANIME:\n")
        top_anime = df.nlargest(10, 'score')[['title', 'score', 'episodes']]
        for idx, (_, row) in enumerate(top_anime.iterrows(), 1):
            f.write(f"  {idx}. {row['title']} - Score: {row['score']}, Episodes: {row['episodes']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print("✓ Saved: summary_statistics.txt")


def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*80)
    print("ANIME DATA VISUALIZATION")
    print("="*80 + "\n")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('anime_recommendation_dataset.csv')
    print(f"Loaded {len(df)} anime entries\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    plot_score_distribution(df)
    plot_episode_distribution(df)
    plot_genre_analysis(df)
    plot_genre_scores(df)
    plot_word_frequency(df)
    plot_score_vs_episodes(df)
    generate_summary_stats(df)
    
    print("\n" + "="*80)
    print("All visualizations generated successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - score_distribution.png")
    print("  - episode_distribution.png")
    print("  - genre_distribution.png")
    print("  - genre_avg_scores.png")
    print("  - word_frequency.png")
    print("  - score_vs_episodes.png")
    print("  - summary_statistics.txt")
    print("\n")


if __name__ == "__main__":
    main()
