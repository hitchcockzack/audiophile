# 🎵 Audiophile - Enhanced Intelligent Music Discovery Platform

Welcome to the next generation of music discovery! Audiophile has evolved from a simple recommendation system into a comprehensive AI-powered music intelligence platform that understands your taste better than you do.

## 🌟 Revolutionary Features

### 🔥 **NEW:** Relative Popularity Analysis
Instead of just showing you globally popular tracks, we analyze popularity **relative to each artist's catalog**. Discover tracks that are:
- Climbing the ranks within an artist's discography
- Recently released with momentum
- Hidden gems with high discovery potential
- Perfect balance of familiar yet fresh

**Example:** Find a track with only 8k plays that's actually the #2 most popular song from an artist whose #1 hit has 12k plays and has been out for a year. Our system identifies this as a rising star worth your attention.

### 🧬 **NEW:** Hyperdimensional Audio Fingerprinting
We extract **300+ unique features** from each song, creating a comprehensive "DNA" profile:

#### Spectral Analysis (50+ features)
- MFCC coefficients with temporal evolution
- Spectral centroid, rolloff, bandwidth, flatness
- Harmonic vs percussive energy distribution
- Spectral complexity and contrast

#### Vocal Characteristics (45+ features)
- Formant frequency analysis
- Vibrato detection and strength
- Vocal range and timbre
- Brightness, roughness, and warmth

#### Harmonic & Tonal (40+ features)
- Chroma features and key estimation
- Chord progression analysis
- Tonnetz harmonic network
- Modulation and key changes

#### Rhythmic Patterns (35+ features)
- Advanced tempo and beat analysis
- Syncopation and groove detection
- Polyrhythmic complexity
- Onset pattern analysis

#### Structural Analysis (25+ features)
- Song section detection
- Repetition and novelty analysis
- Form and arrangement patterns
- Dynamic evolution

#### Psychoacoustic Features (25+ features)
- Bark scale analysis
- Perceived loudness and roughness
- Sharpness and fluctuation strength
- Critical band processing

#### Cultural Markers (15+ features)
- Rhythm pattern analysis for genre/culture
- Pentatonic scale detection
- Microtonal content analysis
- Instrumentation hints

#### Lyrical Analysis (20+ features) *with Genius API*
- Sentiment and emotional content
- Complexity and lexical diversity
- Theme detection and keyword analysis
- Repetition and structure patterns

### 🧠 **NEW:** AI Learning & Feedback System
Our intelligent feedback system continuously adapts to your evolving taste:

- **Real-time Learning:** Algorithm adapts with every piece of feedback
- **Pattern Recognition:** Identifies subtle preference patterns you might not even notice
- **Temporal Evolution:** Tracks how your taste changes over time
- **Context Awareness:** Considers mood, time of day, and listening context
- **Predictive Modeling:** Uses machine learning to predict your rating for new tracks

### 🎯 **NEW:** Confidence Scoring
Only receive recommendations when we're confident they're perfect:
- Minimum 90% confidence threshold
- Multi-factor scoring system
- Quality over quantity approach
- "Divine-level" recommendations only

### 🌐 **NEW:** Web Application Interface
Beautiful, responsive web app that centralizes all features:
- **Real-time Dashboard:** Live updates via WebSocket
- **Interactive Analytics:** Visualize your taste evolution
- **Feedback Collection:** Seamless like/dislike/love system
- **Playlist Creation:** AI-assisted smart playlist generation
- **Discovery Mode:** Venture beyond comfort zone with confidence

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd audiophile

# Install enhanced dependencies
pip install -r requirements_enhanced.txt

# Set up environment variables
cp config/env_template.txt .env
# Edit .env with your API keys
```

### 2. Required API Keys

```bash
# Spotify API (Required)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:5000/callback

# Genius API (Optional - for lyrical analysis)
GENIUS_API_KEY=your_genius_api_key

# Database
DATABASE_PATH=data/audiophile.db

# Web App
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

### 3. Launch the Web Application

```bash
# Start the enhanced web application
python app.py
```

Navigate to `http://localhost:5001` and connect your Spotify account to begin your intelligent music discovery journey!

**Note:** We use port 5001 by default to avoid conflicts with macOS AirPlay on port 5000.

## 🎛️ Advanced Usage

### Command Line Interface

#### Generate Intelligent Recommendations
```bash
# Get personalized recommendations with relative popularity
python get_recommendations.py --enhanced --discovery-mode

# Mood-based recommendations
python get_recommendations.py --mood="energetic" --count=30

# Create discovery playlist with hidden gems
python create_discovery_playlist.py --relative-popularity --hidden-gems
```

#### Analyze Your Music Collection
```bash
# Comprehensive analysis with hyperdimensional fingerprinting
python src/main.py --full-analysis --save-features

# Relative popularity analysis for your tracks
python -c "
from src.recommendations.relative_popularity import RelativePopularityAnalyzer
from src.auth.spotify_auth import get_spotify_client
analyzer = RelativePopularityAnalyzer(get_spotify_client(), None)
# Your analysis code here
"
```

### Web API Endpoints

#### Get Enhanced Recommendations
```bash
curl -X POST http://localhost:5000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "count": 20,
    "type": "discovery",
    "mood": "chill",
    "discovery_mode": true
  }'
```

#### Submit Feedback
```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "track_id": "spotify_track_id",
    "feedback_type": "love",
    "context": {
      "mood": "happy",
      "time_of_day": "evening"
    }
  }'
```

#### Analyze Track
```bash
curl -X POST http://localhost:5000/api/analyze_track \
  -H "Content-Type: application/json" \
  -d '{"track_id": "spotify_track_id"}'
```

## 🔬 Technical Architecture

### Core Components

```
audiophile/
├── src/
│   ├── recommendations/
│   │   ├── recommender.py              # Enhanced recommendation engine
│   │   └── relative_popularity.py      # Relative popularity analysis
│   ├── analysis/
│   │   ├── hyperdimensional_analyzer.py # 300+ feature extraction
│   │   └── audio_intelligence.py       # AI orchestration
│   ├── feedback/
│   │   └── feedback_system.py          # Learning & adaptation
│   ├── auth/
│   │   └── spotify_auth.py             # Authentication
│   ├── database/
│   │   └── models.py                   # Enhanced database schema
│   └── data/
│       └── collector.py                # Data collection
├── templates/                          # Web application UI
├── app.py                             # Flask web application
└── requirements_enhanced.txt          # All dependencies
```

### Algorithm Flow

1. **Data Collection:** Spotify API + Enhanced Metadata
2. **Hyperdimensional Analysis:** 300+ features per track
3. **Relative Popularity Calculation:** Artist-specific metrics
4. **Taste Profile Building:** ML-based user modeling
5. **Intelligent Recommendation:** Multi-factor scoring
6. **Feedback Integration:** Real-time algorithm adaptation
7. **Continuous Learning:** Preference evolution tracking

## 📊 Understanding Your Analytics

### Relative Popularity Metrics
- **Momentum Score:** How fast a track is climbing relative to artist's catalog
- **Discovery Score:** Likelihood of being a hidden gem
- **Relative Position:** Percentile within artist's tracks
- **Rising Star:** High momentum + not yet mainstream
- **Hidden Gem:** Low global popularity + high artist-relative popularity

### Feedback Analytics
- **Recommendation Accuracy:** Percentage of positive feedback
- **Learning Progress:** How well the algorithm knows your taste
- **Preference Stability:** How consistent your taste is over time
- **Discovery Tendency:** Your openness to new/different music

### Audio Features Dashboard
View detailed breakdowns of:
- Spectral characteristics of your preferred music
- Vocal qualities you gravitate toward
- Harmonic preferences and patterns
- Rhythmic complexity preferences
- Emotional content analysis

## 🎯 Pro Tips

### Maximizing Recommendation Quality
1. **Give Honest Feedback:** The system learns from every interaction
2. **Use Discovery Mode:** Gradually expand your musical horizons
3. **Provide Context:** Time of day, mood, and activity matter
4. **Be Patient:** The algorithm improves significantly after 50+ feedback items
5. **Explore Analytics:** Understand your taste to discover new directions

### Finding Hidden Gems
1. Enable **Discovery Mode** in recommendations
2. Sort by **Discovery Score** in relative popularity analysis
3. Look for tracks with high **Momentum Score** but low global popularity
4. Use **Rising Star** filter to find tracks gaining traction

### Training Your Algorithm
1. Rate at least 20-30 tracks to start seeing improvements
2. Provide feedback consistently over time
3. Use the "Train Algorithm" button after major feedback sessions
4. Review your analytics to understand algorithmic adaptations

## 🛠️ Development & Customization

### Adding New Audio Features
```python
# In src/analysis/hyperdimensional_analyzer.py
def _extract_custom_features(self, audio_data: np.ndarray) -> Dict:
    features = {}

    # Your custom feature extraction here
    features['my_custom_feature'] = calculate_my_feature(audio_data)

    return features
```

### Custom Recommendation Algorithms
```python
# In src/recommendations/recommender.py
def get_custom_recommendations(self, params: Dict) -> List[Dict]:
    # Implement your custom recommendation logic
    pass
```

### Database Extensions
```python
# In src/database/models.py
def create_custom_tables(self):
    # Add your custom database tables
    pass
```

## 🔧 Configuration Options

### Audio Analysis Settings
```python
# Hyperdimensional analyzer configuration
SAMPLE_RATE = 22050
HOP_LENGTH = 512
FRAME_LENGTH = 2048
N_MFCC = 20
N_CHROMA = 12
```

### Recommendation Parameters
```python
# Default recommendation weights
FEATURE_WEIGHTS = {
    'relative_popularity': 0.3,
    'audio_similarity': 0.4,
    'user_preference': 0.3
}

CONFIDENCE_THRESHOLD = 0.9
DISCOVERY_BOOST = 0.2
```

### Feedback Learning Settings
```python
# Learning algorithm parameters
LEARNING_RATE = 0.01
ADAPTATION_THRESHOLD = 50  # Feedback items before adaptation
PREFERENCE_DECAY = 0.95    # How quickly old preferences fade
```

## 📈 Performance Optimization

### For Large Music Collections
- Enable **batch processing** for analysis
- Use **caching** for frequently accessed data
- Implement **background workers** for heavy computations
- Consider **database indexing** for faster queries

### For Real-time Recommendations
- **Pre-compute** similarity matrices
- Use **async processing** for web requests
- Implement **recommendation caching**
- Enable **WebSocket** for instant updates

## 🤝 Contributing

We welcome contributions to make Audiophile even more intelligent!

### Priority Areas
1. **Enhanced Audio Analysis:** New feature extraction methods
2. **Advanced ML Models:** Better prediction algorithms
3. **UI/UX Improvements:** Better user experience
4. **Integration APIs:** Support for more music platforms
5. **Performance Optimizations:** Faster analysis and recommendations

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_enhanced.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 🎵 The Science Behind the Magic

### Why 300+ Features Matter
Traditional recommendation systems use 10-20 basic audio features. We extract 300+ because:
- **Musical complexity** requires nuanced analysis
- **Personal taste** involves subtle preferences traditional systems miss
- **Emotional resonance** depends on micro-details in sound
- **Cultural context** requires deep understanding of musical patterns

### Relative Popularity Innovation
Most systems recommend globally popular tracks, creating echo chambers. Our relative popularity analysis:
- **Identifies rising tracks** within artist catalogs
- **Discovers hidden gems** before they become mainstream
- **Balances familiarity with novelty**
- **Respects artist evolution** and catalog depth

### AI Learning Architecture
Our feedback system uses advanced machine learning:
- **Random Forest** for preference prediction
- **Feature importance** analysis for taste understanding
- **Temporal modeling** for preference evolution
- **Context integration** for situational recommendations

## 🌟 What Makes This Special

Audiophile isn't just another music recommendation system. It's a comprehensive music intelligence platform that:

1. **Understands music at a molecular level** with 300+ features
2. **Learns your taste better than you know it** with advanced AI
3. **Discovers hidden gems** through relative popularity analysis
4. **Adapts continuously** to your evolving preferences
5. **Provides context and reasoning** for every recommendation
6. **Respects musical diversity** and artist evolution

## 🎶 Start Your Enhanced Music Discovery Journey

Ready to experience music discovery like never before?

```bash
python app.py
```

Navigate to `http://localhost:5000`, connect your Spotify account, and prepare to discover music that truly moves you.

---

**"Music is the universal language of mankind, and now we speak it fluently."** - Audiophile Team

*Built with ❤️ and powered by cutting-edge AI*
