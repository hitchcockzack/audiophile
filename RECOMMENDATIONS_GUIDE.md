# 🎵 Getting Music Recommendations

You now have multiple ways to get music recommendations based on your Spotify data!

## Quick Start (Easiest Way)

Run the simple recommendation script:

```bash
python get_recommendations.py
```

This will give you 4 options:

1. **Spotify API recommendations** (🚀 Fastest)
   - Uses your top tracks as seeds
   - Gets recommendations directly from Spotify
   - Works immediately with your CSV data

2. **Advanced audio feature analysis**
   - Uses deep audio analysis if available
   - Requires running the full analysis first
   - Most sophisticated recommendations

3. **Simple content-based filtering**
   - Analyzes your favorite artists and popularity preferences
   - Searches for similar tracks
   - Works with basic metadata

4. **All methods combined**
   - Runs all recommendation types
   - Gives you the most comprehensive results

## Advanced Options

### Full Audio Intelligence Analysis

For the most sophisticated recommendations, run the complete analysis:

```bash
python src/main.py
```

This will:
- 🔬 Download audio previews and extract technical features
- 🎯 Analyze your music taste profile in detail
- 🤖 Generate AI-powered recommendations
- 🎵 Create smart playlists

### Manual Recommendation Engine

If you want to work with the recommendation engine directly:

```python
# Example usage
from src.recommendations.recommender import MusicRecommender
import pandas as pd

# Load your data
features_df = pd.read_csv("data/audio_features.csv")  # If you have features
tracks_df = pd.read_csv("data/user_data_20250725_125118.csv")

# Initialize recommender
recommender = MusicRecommender(features_df, tracks_df)

# Get recommendations for a specific track
recommendations = recommender.find_similar_tracks("track_id_here", n_recommendations=10)

# Get mood-based recommendations
chill_tracks = recommender.get_mood_based_recommendations("chill", n_recommendations=15)
energetic_tracks = recommender.get_mood_based_recommendations("energetic", n_recommendations=15)

# Create a playlist from multiple seed tracks
playlist = recommender.create_playlist_recommendations(
    seed_tracks=["track1", "track2", "track3"],
    playlist_length=25,
    diversity_factor=0.3
)
```

## Your Data Summary

From `data/user_data_20250725_125118.csv`:
- 📀 **195 tracks** collected
- 🎤 **Various artists** including Jas Von, Jessie Murph, Chase Atlantic, Juice WRLD
- 📊 **Mix of sources**: top tracks, saved tracks, recently played
- 🎵 **Genres**: Hip-hop, pop, alternative, country, rock

## Recommendation Types Available

### 1. Content-Based Filtering
- Analyzes audio features (tempo, energy, valence, etc.)
- Finds tracks with similar characteristics
- Works great for discovering music with similar "feel"

### 2. Collaborative Filtering
- Uses patterns from other users with similar taste
- Powered by Spotify's recommendation API
- Great for finding popular tracks you might like

### 3. Hybrid Approach
- Combines multiple recommendation strategies
- Balances similarity with diversity
- Creates well-rounded playlists

### 4. Mood-Based Recommendations
- **Energetic**: High energy, fast tempo, danceable
- **Chill**: Low energy, slower tempo, acoustic
- **Happy**: High valence, upbeat
- **Focus**: Instrumental, low speechiness
- **Party**: High danceability and energy

## Tips for Better Recommendations

1. **Run the full analysis** for best results:
   ```bash
   python src/main.py
   ```

2. **Use multiple seed tracks** when creating playlists

3. **Experiment with different moods** to discover new music

4. **Save recommendations** to CSV files for later reference

5. **Provide feedback** by saving tracks you like to improve future recommendations

## Troubleshooting

### "No audio features found"
- Run the full analysis: `python src/main.py`
- This downloads audio previews and extracts features

### Spotify authentication issues
- Check your `.env` file has correct credentials
- Make sure your Spotify app is properly configured

### CSV loading errors
- Ensure the CSV file path is correct
- Check that the CSV file has the expected columns

## Next Steps

1. **Try the quick recommendations**: `python get_recommendations.py`
2. **Run full analysis** for advanced features: `python src/main.py`
3. **Explore the recommendation engine** with different parameters
4. **Create custom playlists** based on your mood or activity

Happy music discovery! 🎶
