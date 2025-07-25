# Audiophile 🎵

An intelligent Spotify playlist generator that creates personalized recommendations based on the actual audio features of your favorite songs, not just collaborative filtering.

## 🎯 Project Vision

Instead of relying on "users who liked X also liked Y", Audiophile analyzes the sonic DNA of your favorite tracks (tempo, energy, valence, etc.) to find new music that genuinely matches your sound preferences.

## ✨ Features

- **Audio Feature Analysis**: Deep dive into the musical characteristics of your favorite songs
- **Personalized Taste Profile**: Build a unique sound signature based on your listening habits
- **Smart Recommendations**: Find new music that matches your audio preferences
- **Daily Playlist Updates**: Automatically discover fresh tracks every day
- **Context Awareness**: Factor in weather, time of day, and seasonal preferences (coming soon)

## 🏗️ MVP Goals

1. **Spotify Integration**: OAuth authentication and API access
2. **Data Collection**: Fetch user's top tracks and listening history
3. **Feature Extraction**: Analyze audio characteristics (tempo, energy, valence, etc.)
4. **Taste Modeling**: Create personalized sound profile using clustering
5. **Recommendation Engine**: Find matching tracks using Spotify's recommendation API
6. **Playlist Management**: Auto-create and update "Sound Mirror" playlist
7. **Automation**: Daily scheduled updates

## 🛠️ Tech Stack

- **Backend**: Python with `spotipy` library
- **Data Processing**: pandas, numpy, scikit-learn
- **Scheduler**: GitHub Actions / cron jobs
- **Storage**: SQLite (local) → PostgreSQL (production)
- **Environment**: Virtual environment with requirements.txt

## 🚀 Getting Started

1. **Set up Spotify Developer Account**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create an app to get Client ID and Client Secret
   - Add redirect URI: `http://localhost:8888/callback`

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your Spotify credentials to .env
   ```

4. **Run the Application**
   ```bash
   python src/main.py
   ```

## 📁 Project Structure

```
audiophile/
├── src/
│   ├── auth/           # Spotify authentication
│   ├── data/           # Data collection and storage
│   ├── analysis/       # Audio feature analysis
│   ├── recommendations/ # Recommendation engine
│   ├── playlists/      # Playlist management
│   └── main.py         # Application entry point
├── config/             # Configuration files
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎵 How It Works

1. **Authenticate** with your Spotify account
2. **Analyze** your top 50-100 favorite tracks
3. **Extract** audio features (energy, valence, tempo, etc.)
4. **Build** your unique sound profile
5. **Search** for new tracks matching your profile
6. **Create** daily updated playlists with fresh discoveries

## 🎯 Roadmap

- [ ] **Phase 1**: Core MVP functionality
- [ ] **Phase 2**: Web dashboard for insights
- [ ] **Phase 3**: Context-aware recommendations (weather, time, mood)
- [ ] **Phase 4**: Machine learning improvements
- [ ] **Phase 5**: Social features and sharing

## 📈 Business Potential

- **Personal Tool**: Perfect for music discovery
- **Portfolio Project**: Showcase technical skills
- **SaaS Product**: Subscription service for advanced features
- **API Service**: Provide recommendation engine to other apps

## 🤝 Contributing

This is currently a personal MVP project, but contributions and feedback are welcome!

## 📝 License

MIT License - feel free to use this for your own projects and improvements.
