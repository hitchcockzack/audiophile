"""
Database Models and Schema

Defines the SQLite database structure for storing:
- User profiles and preferences
- Track information and metadata
- Audio features and analysis data
- Recommendation history
- Playlist data
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class AudiophileDatabase:
    """Handles all database operations for the Audiophile application."""

    def __init__(self, db_path: str = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (defaults to env var)
        """
        if db_path is None:
            db_path = os.getenv('DATABASE_PATH', 'data/audiophile.db')

        self.db_path = db_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._create_tables()
        print(f"✅ Database initialized: {db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn

    def _create_tables(self):
        """Create all necessary database tables."""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spotify_user_id TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    email TEXT,
                    country TEXT,
                    followers_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Tracks table - stores basic track information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY,  -- Spotify track ID
                    name TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    album TEXT,
                    popularity INTEGER,
                    duration_ms INTEGER,
                    explicit BOOLEAN,
                    release_date TEXT,
                    uri TEXT,
                    external_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Audio features table - stores Spotify audio analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_features (
                    track_id TEXT PRIMARY KEY,
                    danceability REAL,
                    energy REAL,
                    key INTEGER,
                    loudness REAL,
                    mode INTEGER,
                    speechiness REAL,
                    acousticness REAL,
                    instrumentalness REAL,
                    liveness REAL,
                    valence REAL,
                    tempo REAL,
                    duration_ms INTEGER,
                    time_signature INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            ''')

            # User track interactions - tracks user's relationship with songs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    source TEXT,  -- top_tracks_short, saved_tracks, etc.
                    interaction_type TEXT,  -- played, saved, top_track
                    interaction_date TIMESTAMP,
                    play_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id),
                    UNIQUE(user_id, track_id, source)
                )
            ''')

            # User taste profiles - computed preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS taste_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    profile_name TEXT,  -- main, workout, chill, etc.
                    avg_danceability REAL,
                    avg_energy REAL,
                    avg_valence REAL,
                    avg_tempo REAL,
                    avg_acousticness REAL,
                    avg_instrumentalness REAL,
                    avg_speechiness REAL,
                    avg_liveness REAL,
                    avg_loudness REAL,
                    preferred_keys TEXT,  -- JSON array of preferred keys
                    preferred_modes TEXT,  -- JSON array of preferred modes
                    track_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Recommendations table - stores generated recommendations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    similarity_score REAL,
                    recommendation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    recommended_by TEXT,  -- algorithm version/type
                    context_data TEXT,  -- JSON with context (weather, time, etc.)
                    user_feedback INTEGER,  -- -1 dislike, 0 neutral, 1 like
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            ''')

            # Playlists table - tracks created playlists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS playlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spotify_playlist_id TEXT UNIQUE,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    public BOOLEAN DEFAULT FALSE,
                    track_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Playlist tracks - many-to-many relationship
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS playlist_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    playlist_id INTEGER NOT NULL,
                    track_id TEXT NOT NULL,
                    position INTEGER,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (playlist_id) REFERENCES playlists (id),
                    FOREIGN KEY (track_id) REFERENCES tracks (id),
                    UNIQUE(playlist_id, track_id)
                )
            ''')

            # Data collection logs - track collection history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    collection_type TEXT,  -- full_profile, incremental, etc.
                    tracks_collected INTEGER,
                    tracks_with_features INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT DEFAULT 'completed',  -- completed, failed, partial
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_tracks_user_id ON user_tracks(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_tracks_track_id ON user_tracks(track_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_score ON recommendations(similarity_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_popularity ON tracks(popularity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_features_energy ON audio_features(energy)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_features_valence ON audio_features(valence)')

            conn.commit()
            print("✅ Database tables created successfully")

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to create database tables: {e}")
        finally:
            conn.close()

    def save_user_profile(self, user_data: Dict) -> str:
        """
        Save or update user profile.

        Args:
            user_data: User information from Spotify API

        Returns:
            User ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles
                (spotify_user_id, display_name, email, country, followers_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_data['id'],
                user_data.get('display_name'),
                user_data.get('email'),
                user_data.get('country'),
                user_data.get('followers', {}).get('total', 0),
                datetime.now()
            ))

            conn.commit()
            print(f"✅ Saved user profile: {user_data.get('display_name', user_data['id'])}")
            return user_data['id']

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to save user profile: {e}")
        finally:
            conn.close()

    def save_track(self, track_data: Dict) -> str:
        """
        Save track information to database.

        Args:
            track_data: Track information from Spotify API

        Returns:
            Track ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO tracks
                (id, name, artist, album, popularity, duration_ms, explicit,
                 release_date, uri, external_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_data['id'],
                track_data['name'],
                track_data['artist'],
                track_data['album'],
                track_data['popularity'],
                track_data['duration_ms'],
                track_data['explicit'],
                track_data['release_date'],
                track_data['uri'],
                track_data['external_url']
            ))

            conn.commit()
            return track_data['id']

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to save track {track_data.get('name', 'Unknown')}: {e}")
        finally:
            conn.close()

    def save_audio_features(self, features_data: Dict) -> str:
        """
        Save audio features to database.

        Args:
            features_data: Audio features from Spotify API

        Returns:
            Track ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO audio_features
                (track_id, danceability, energy, key, loudness, mode, speechiness,
                 acousticness, instrumentalness, liveness, valence, tempo,
                 duration_ms, time_signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features_data['id'],
                features_data.get('danceability'),
                features_data.get('energy'),
                features_data.get('key'),
                features_data.get('loudness'),
                features_data.get('mode'),
                features_data.get('speechiness'),
                features_data.get('acousticness'),
                features_data.get('instrumentalness'),
                features_data.get('liveness'),
                features_data.get('valence'),
                features_data.get('tempo'),
                features_data.get('duration_ms'),
                features_data.get('time_signature')
            ))

            conn.commit()
            return features_data['id']

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to save audio features for track {features_data.get('id', 'Unknown')}: {e}")
        finally:
            conn.close()

    def save_user_track_interaction(self, user_id: str, track_id: str, source: str,
                                  interaction_type: str = 'collected') -> None:
        """
        Save user's interaction with a track.

        Args:
            user_id: Spotify user ID
            track_id: Spotify track ID
            source: Source of the interaction (top_tracks_short, saved_tracks, etc.)
            interaction_type: Type of interaction
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO user_tracks
                (user_id, track_id, source, interaction_type, interaction_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, track_id, source, interaction_type, datetime.now()))

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to save user track interaction: {e}")
        finally:
            conn.close()

    def get_user_tracks_with_features(self, user_id: str, limit: int = None) -> List[Dict]:
        """
        Get user's tracks with audio features.

        Args:
            user_id: Spotify user ID
            limit: Maximum number of tracks to return

        Returns:
            List of tracks with audio features
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            query = '''
                SELECT t.*, af.*, ut.source, ut.interaction_type, ut.interaction_date
                FROM tracks t
                JOIN audio_features af ON t.id = af.track_id
                JOIN user_tracks ut ON t.id = ut.track_id
                WHERE ut.user_id = ?
                ORDER BY ut.interaction_date DESC
            '''

            if limit:
                query += f' LIMIT {limit}'

            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            tracks = []
            for row in rows:
                track_dict = dict(row)
                tracks.append(track_dict)

            return tracks

        except Exception as e:
            raise Exception(f"Failed to get user tracks with features: {e}")
        finally:
            conn.close()

    def log_collection_session(self, user_id: str, collection_type: str,
                             tracks_collected: int, tracks_with_features: int,
                             start_time: datetime, status: str = 'completed',
                             error_message: str = None) -> int:
        """
        Log a data collection session.

        Args:
            user_id: Spotify user ID
            collection_type: Type of collection performed
            tracks_collected: Number of tracks collected
            tracks_with_features: Number of tracks with audio features
            start_time: When collection started
            status: Status of collection
            error_message: Error message if failed

        Returns:
            Log entry ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO collection_logs
                (user_id, collection_type, tracks_collected, tracks_with_features,
                 start_time, end_time, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, collection_type, tracks_collected, tracks_with_features,
                start_time, datetime.now(), status, error_message
            ))

            log_id = cursor.lastrowid
            conn.commit()

            print(f"✅ Logged collection session: {tracks_collected} tracks, {status}")
            return log_id

        except Exception as e:
            conn.rollback()
            raise Exception(f"Failed to log collection session: {e}")
        finally:
            conn.close()

    def get_database_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            stats = {}

            # Count tables
            tables = ['tracks', 'audio_features', 'user_tracks', 'taste_profiles',
                     'recommendations', 'playlists', 'collection_logs']

            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]

            # Get latest collection
            cursor.execute('''
                SELECT MAX(created_at) FROM collection_logs
                WHERE status = 'completed'
            ''')
            stats['latest_collection'] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            raise Exception(f"Failed to get database stats: {e}")
        finally:
            conn.close()


def initialize_database(db_path: str = None) -> AudiophileDatabase:
    """
    Initialize the Audiophile database.

    Args:
        db_path: Path to database file

    Returns:
        Database instance
    """
    return AudiophileDatabase(db_path)


if __name__ == "__main__":
    """Test database setup."""
    try:
        print("🗄️  Testing database setup...")

        # Initialize database
        db = initialize_database("data/test_audiophile.db")

        # Get stats
        stats = db.get_database_stats()
        print(f"📊 Database stats: {stats}")

        print("✅ Database setup test completed successfully!")

        # Clean up test database
        import os
        if os.path.exists("data/test_audiophile.db"):
            os.remove("data/test_audiophile.db")
            print("🗑️  Cleaned up test database")

    except Exception as e:
        print(f"❌ Database setup test failed: {e}")
