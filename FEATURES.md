# 🎵 Audiophile - Feature Overview

## Core Enhanced Features

### 🔥 **Relative Popularity Analysis**
- Analyzes popularity **within artist catalogs** vs global metrics
- Identifies rising tracks (8k plays climbing vs 12k established hit)
- Detects hidden gems and momentum patterns
- Discovery scoring for tracks gaining traction

### 🧬 **Hyperdimensional Audio Analysis (300+ Features)**
- **Spectral Analysis**: MFCC, centroid, rolloff, bandwidth, contrast
- **Vocal Characteristics**: Formant analysis, vibrato, timbre, brightness
- **Harmonic Structure**: Chroma, key estimation, chord progressions
- **Rhythmic Patterns**: Tempo analysis, syncopation, groove detection
- **Structural Analysis**: Song sections, repetition, dynamic evolution
- **Psychoacoustic**: Bark scale, perceived roughness, sharpness
- **Cultural Markers**: Genre patterns, pentatonic detection, instrumentation
- **Lyrical Analysis**: Sentiment, complexity, emotional content (with Genius API)

### 🧠 **AI Learning & Feedback System**
- Real-time algorithm adaptation from user feedback
- Pattern recognition for subtle preference detection
- Preference evolution tracking over time
- Context-aware learning (mood, time, activity)
- Machine learning models for preference prediction
- Confidence scoring (90%+ threshold for recommendations)

### 🎭 **Magic Recommendation Narrator**
- Contextual explanations for every recommendation
- Sonic DNA connection stories
- Emotional journey descriptions
- Discovery insights and listening instructions
- Mystery and purpose in every suggestion

### 🌐 **Web Application Interface**
- Real-time dashboard with WebSocket updates
- Interactive feedback collection (like/dislike/love)
- Smart playlist creation with diversity controls
- Discovery mode for comfort zone expansion
- Analytics and taste profile visualization
- Responsive design with Spotify-inspired aesthetic

## Advanced Capabilities

### 📊 **Intelligence Features**
- Multi-algorithm recommendation engine
- Cluster-based similarity detection
- Mood-based track filtering
- Feature-weighted personalization
- Temporal pattern analysis

### 🎯 **Discovery Engine**
- Hidden gem identification
- Rising star detection
- Artist catalog deep-dive analysis
- Momentum scoring for trending tracks
- Cultural and genre pattern recognition

### 📈 **Analytics & Insights**
- Taste profile evolution tracking
- Recommendation accuracy metrics
- Preference stability analysis
- Discovery tendency scoring
- Feature importance visualization

### 🎵 **Smart Playlists**
- AI-curated balanced playlists
- Customizable diversity factors
- Smooth transition optimization
- Context-aware curation
- Dynamic length adjustment

## Technical Architecture

### **Core Modules**
- `relative_popularity.py` - Artist-relative analysis
- `hyperdimensional_analyzer.py` - 300+ feature extraction
- `feedback_system.py` - AI learning & adaptation
- `magic_narrator.py` - Contextual explanations
- `audio_intelligence.py` - AI orchestration

### **Web Interface**
- Flask application with real-time features
- WebSocket integration for instant updates
- Beautiful dark UI with responsive design
- RESTful API endpoints for all features

### **Database**
- Enhanced schema for feedback tracking
- Preference evolution storage
- Algorithm adaptation logging
- Comprehensive audio feature storage

## Usage Modes

### **Web Application** (Recommended)
```bash
python app.py
# Navigate to http://localhost:5001
```

### **Command Line**
```bash
# Enhanced recommendations
python get_recommendations.py --enhanced --discovery-mode

# Smart playlist creation
python create_discovery_playlist.py --relative-popularity
```

### **API Integration**
- POST `/api/recommendations` - Get enhanced recommendations
- POST `/api/feedback` - Submit user feedback
- POST `/api/analyze_track` - Hyperdimensional analysis
- POST `/api/relative_popularity` - Artist-relative metrics

## Key Innovations

1. **Relative Popularity**: First system to analyze artist-internal track rankings
2. **300+ Features**: Most comprehensive audio analysis in music recommendation
3. **Contextual AI**: Learns and adapts with explanatory narratives
4. **Discovery Focus**: Finds hidden gems before they become mainstream
5. **Confidence Scoring**: Only recommends when algorithm is highly confident

## Dependencies

- **Core**: spotipy, pandas, numpy, scikit-learn
- **Audio**: librosa, essentia (optional), madmom (optional)
- **Web**: Flask, Flask-SocketIO
- **ML**: textblob, nltk (for lyrics)
- **Utilities**: python-dateutil, requests, python-dotenv

---

*Built for true music discovery - finding songs that move your soul, not just fill silence.*
