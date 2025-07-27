"""
Audio Intelligence Integration Module

Orchestrates the complete audio analysis and recommendation pipeline:
1. Collects Spotify track data with preview URLs
2. Downloads and analyzes audio using Essentia/librosa
3. Stores comprehensive audio features in database
4. Generates intelligent recommendations using multiple algorithms
5. Creates personalized playlists and taste profiles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import json

from src.analysis.audio_analyzer import AudioAnalyzer
from src.recommendations.recommender import MusicRecommender
from src.database.models import AudiophileDatabase


class AudioIntelligenceEngine:
    """Main orchestrator for audio analysis and intelligent recommendations."""

    def __init__(self, spotify_client, database: AudiophileDatabase):
        """
        Initialize the audio intelligence engine.

        Args:
            spotify_client: Authenticated Spotify client
            database: Database instance for storing results
        """
        self.spotify = spotify_client
        self.database = database
        self.audio_analyzer = None
        self.recommender = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize audio analyzer and other components."""
        try:
            self.audio_analyzer = AudioAnalyzer()
            print("✅ Audio Intelligence Engine initialized")
        except Exception as e:
            print(f"⚠️  Error initializing audio analyzer: {e}")
            print("   Some features may be limited without audio analysis libraries")

    def enrich_tracks_with_previews(self, tracks: List[Dict]) -> List[Dict]:
        """
        Enrich track data with preview URLs from Spotify.

        Args:
            tracks: List of track dictionaries

        Returns:
            List of tracks with preview URLs added
        """
        enriched_tracks = []

        print("🔗 Enriching tracks with preview URLs...")

        for i, track in enumerate(tracks):
            try:
                # Get full track info from Spotify API
                track_id = track.get('id')
                if track_id:
                    spotify_track = self.spotify.track(track_id)

                    # Add preview URL if available
                    track['preview_url'] = spotify_track.get('preview_url')
                    track['spotify_url'] = spotify_track['external_urls']['spotify']

                    # Add additional metadata
                    if 'album' in spotify_track:
                        track['album_image'] = spotify_track['album']['images'][0]['url'] if spotify_track['album']['images'] else None
                        track['release_date'] = spotify_track['album']['release_date']

                    enriched_tracks.append(track)

                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{len(tracks)} tracks")

            except Exception as e:
                print(f"⚠️  Error enriching track {track.get('name', 'Unknown')}: {e}")
                enriched_tracks.append(track)  # Add without preview URL

        preview_count = sum(1 for t in enriched_tracks if t.get('preview_url'))
        print(f"✅ Enriched {len(enriched_tracks)} tracks ({preview_count} with preview URLs)")

        return enriched_tracks

    def analyze_user_music_collection(self, user_data: Dict,
                                    max_analysis_tracks: int = 50) -> Dict:
        """
        Perform comprehensive analysis of user's music collection.

        Args:
            user_data: User data from Spotify collector
            max_analysis_tracks: Maximum tracks to analyze with audio processing

        Returns:
            Dictionary with analysis results
        """
        print("🎼 Starting comprehensive music collection analysis...")

        results = {
            'user_info': user_data.get('user_info', {}),
            'analysis_timestamp': datetime.now().isoformat(),
            'total_tracks': len(user_data.get('tracks', [])),
            'tracks_analyzed': 0,
            'audio_features_extracted': 0,
            'taste_profile': {},
            'track_clusters': {},
            'recommendations': {},
            'analysis_summary': {}
        }

        tracks = user_data.get('tracks', [])
        if not tracks:
            print("❌ No tracks found in user data")
            return results

        # Step 1: Enrich tracks with preview URLs
        enriched_tracks = self.enrich_tracks_with_previews(tracks)

        # Step 2: Audio analysis on tracks with previews
        if self.audio_analyzer:
            tracks_with_previews = [t for t in enriched_tracks if t.get('preview_url')]

            if tracks_with_previews:
                print(f"🎵 Found {len(tracks_with_previews)} tracks with preview URLs")

                # Limit analysis for performance
                analysis_tracks = tracks_with_previews[:max_analysis_tracks]
                print(f"🔬 Analyzing {len(analysis_tracks)} tracks (limited for performance)")

                # Batch analyze audio features
                audio_features = self.audio_analyzer.batch_analyze(analysis_tracks)

                # Store audio features in database
                for features in audio_features:
                    self.database.save_enhanced_audio_features(features)

                results['audio_features_extracted'] = len(audio_features)
                results['tracks_analyzed'] = len(analysis_tracks)

                print(f"✅ Extracted audio features for {len(audio_features)} tracks")

                # Step 3: Create recommendation engine
                if audio_features:
                    features_df = pd.DataFrame(audio_features)
                    tracks_df = pd.DataFrame(enriched_tracks)

                    self.recommender = MusicRecommender(features_df, tracks_df)

                    # Step 4: Analyze taste profile
                    user_track_ids = [t['id'] for t in enriched_tracks]
                    taste_profile = self.recommender.analyze_user_taste_profile(user_track_ids)
                    results['taste_profile'] = taste_profile

                    # Store taste profile in database
                    user_id = user_data.get('user_info', {}).get('id')
                    if user_id and taste_profile:
                        self.database.save_taste_profile(user_id, taste_profile)

                    print("✅ Generated taste profile")

            else:
                print("⚠️  No tracks with preview URLs found for audio analysis")
        else:
            print("⚠️  Audio analyzer not available - skipping audio feature extraction")

        # Step 5: Generate analysis summary
        results['analysis_summary'] = self._generate_analysis_summary(enriched_tracks, results)

        print("🎯 Music collection analysis complete!")
        return results

    def generate_intelligent_recommendations(self, user_tracks: List[str],
                                           recommendation_types: List[str] = None,
                                           n_recommendations: int = 20) -> Dict:
        """
        Generate intelligent recommendations using multiple algorithms.

        Args:
            user_tracks: List of user's track IDs
            recommendation_types: Types of recommendations to generate
            n_recommendations: Number of recommendations per type

        Returns:
            Dictionary with different types of recommendations
        """
        if not self.recommender:
            print("❌ Recommender not initialized - run audio analysis first")
            return {}

        if recommendation_types is None:
            recommendation_types = ['similar', 'cluster', 'mood', 'personalized']

        print("🤖 Generating intelligent recommendations...")

        recommendations = {}

        try:
            # Similar tracks (content-based)
            if 'similar' in recommendation_types and user_tracks:
                print("   🔍 Finding similar tracks...")
                similar_recs = []
                for track_id in user_tracks[:3]:  # Use top 3 as seeds
                    similar = self.recommender.find_similar_tracks(
                        track_id, n_recommendations=n_recommendations//3
                    )
                    similar_recs.extend(similar)

                # Remove duplicates
                seen = set()
                unique_similar = []
                for rec in similar_recs:
                    if rec['id'] not in seen:
                        unique_similar.append(rec)
                        seen.add(rec['id'])

                recommendations['similar_tracks'] = unique_similar[:n_recommendations]

            # Cluster-based recommendations
            if 'cluster' in recommendation_types and user_tracks:
                print("   🎯 Finding cluster-based recommendations...")
                cluster_recs = []
                for track_id in user_tracks[:2]:
                    cluster = self.recommender.get_cluster_recommendations(
                        track_id, n_recommendations=n_recommendations//2
                    )
                    cluster_recs.extend(cluster)

                recommendations['cluster_based'] = cluster_recs[:n_recommendations]

            # Mood-based recommendations
            if 'mood' in recommendation_types:
                print("   😊 Generating mood-based recommendations...")
                moods = ['energetic', 'chill', 'happy', 'party']
                mood_recs = {}
                for mood in moods:
                    mood_tracks = self.recommender.get_mood_based_recommendations(
                        mood, n_recommendations=n_recommendations//2
                    )
                    mood_recs[mood] = mood_tracks

                recommendations['mood_based'] = mood_recs

            # Personalized recommendations
            if 'personalized' in recommendation_types and user_tracks:
                print("   👤 Creating personalized recommendations...")
                personalized = self.recommender.get_personalized_recommendations(
                    user_tracks, n_recommendations=n_recommendations
                )
                recommendations['personalized'] = personalized

            print(f"✅ Generated {len(recommendations)} types of recommendations")

            # Store recommendations in database
            user_id = user_tracks[0] if user_tracks else 'unknown'  # Simplified for demo
            self._store_recommendations(user_id, recommendations)

            return recommendations

        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return {}

    def create_smart_playlist(self, seed_tracks: List[str],
                            playlist_name: str = "AI Generated Playlist",
                            playlist_length: int = 30,
                            diversity_factor: float = 0.3) -> Dict:
        """
        Create an intelligent playlist using audio analysis.

        Args:
            seed_tracks: List of seed track IDs
            playlist_name: Name for the playlist
            playlist_length: Target playlist length
            diversity_factor: Balance between similarity and diversity

        Returns:
            Dictionary with playlist information
        """
        if not self.recommender:
            print("❌ Recommender not initialized")
            return {}

        print(f"🎵 Creating smart playlist: {playlist_name}")

        try:
            # Generate balanced playlist
            playlist_tracks = self.recommender.create_playlist_recommendations(
                seed_tracks,
                playlist_length=playlist_length,
                diversity_factor=diversity_factor
            )

            playlist_info = {
                'name': playlist_name,
                'description': f'AI-generated playlist based on {len(seed_tracks)} seed tracks',
                'tracks': playlist_tracks,
                'length': len(playlist_tracks),
                'seed_tracks': seed_tracks,
                'diversity_factor': diversity_factor,
                'created_at': datetime.now().isoformat(),
                'algorithm': 'audio_intelligence_v1'
            }

            print(f"✅ Created playlist with {len(playlist_tracks)} tracks")
            return playlist_info

        except Exception as e:
            print(f"❌ Error creating smart playlist: {e}")
            return {}

    def analyze_track_compatibility(self, track1_id: str, track2_id: str) -> Dict:
        """
        Analyze how compatible two tracks are for mixing/playlist placement.

        Args:
            track1_id: First track ID
            track2_id: Second track ID

        Returns:
            Dictionary with compatibility analysis
        """
        if not self.recommender:
            return {}

        try:
            # Get track features
            features_df = self.recommender.features_df

            track1_features = features_df[features_df['track_id'] == track1_id]
            track2_features = features_df[features_df['track_id'] == track2_id]

            if len(track1_features) == 0 or len(track2_features) == 0:
                return {'error': 'Track features not found'}

            track1_idx = track1_features.index[0]
            track2_idx = track2_features.index[0]

            # Calculate overall similarity
            similarity = self.recommender.similarity_matrix[track1_idx][track2_idx]

            # Analyze specific compatibility factors
            t1 = track1_features.iloc[0]
            t2 = track2_features.iloc[0]

            compatibility = {
                'overall_similarity': float(similarity),
                'tempo_compatibility': self._analyze_tempo_compatibility(t1, t2),
                'key_compatibility': self._analyze_key_compatibility(t1, t2),
                'energy_compatibility': self._analyze_energy_compatibility(t1, t2),
                'mood_compatibility': self._analyze_mood_compatibility(t1, t2),
                'recommendation': self._get_compatibility_recommendation(similarity)
            }

            return compatibility

        except Exception as e:
            print(f"❌ Error analyzing track compatibility: {e}")
            return {}

    def _analyze_tempo_compatibility(self, track1, track2) -> Dict:
        """Analyze tempo compatibility between two tracks."""
        tempo1 = track1.get('tempo_essentia', track1.get('tempo_librosa', 0))
        tempo2 = track2.get('tempo_essentia', track2.get('tempo_librosa', 0))

        if tempo1 == 0 or tempo2 == 0:
            return {'compatible': False, 'reason': 'Tempo data missing'}

        tempo_diff = abs(tempo1 - tempo2)
        tempo_ratio = max(tempo1, tempo2) / min(tempo1, tempo2)

        # Check for harmonic relationships (2:1, 3:2, etc.)
        harmonic_ratios = [1, 2, 1.5, 0.75, 0.67, 1.33]
        is_harmonic = any(abs(tempo_ratio - ratio) < 0.1 for ratio in harmonic_ratios)

        return {
            'tempo1': tempo1,
            'tempo2': tempo2,
            'difference': tempo_diff,
            'ratio': tempo_ratio,
            'compatible': tempo_diff < 20 or is_harmonic,
            'harmonic_relationship': is_harmonic
        }

    def _analyze_key_compatibility(self, track1, track2) -> Dict:
        """Analyze key compatibility between two tracks."""
        key1 = track1.get('key')
        key2 = track2.get('key')

        if not key1 or not key2:
            return {'compatible': None, 'reason': 'Key data missing'}

        # Simplified key compatibility (same key or related keys)
        compatible_keys = {
            'C': ['C', 'G', 'F', 'Am', 'Em', 'Dm'],
            'G': ['G', 'D', 'C', 'Em', 'Bm', 'Am'],
            # Add more key relationships as needed
        }

        is_compatible = key2 in compatible_keys.get(key1, [key1])

        return {
            'key1': key1,
            'key2': key2,
            'compatible': is_compatible,
            'same_key': key1 == key2
        }

    def _analyze_energy_compatibility(self, track1, track2) -> Dict:
        """Analyze energy level compatibility."""
        energy1 = track1.get('energy', 0)
        energy2 = track2.get('energy', 0)

        energy_diff = abs(energy1 - energy2)

        return {
            'energy1': energy1,
            'energy2': energy2,
            'difference': energy_diff,
            'compatible': energy_diff < 0.3  # Similar energy levels
        }

    def _analyze_mood_compatibility(self, track1, track2) -> Dict:
        """Analyze mood compatibility using valence."""
        valence1 = track1.get('valence', 0)
        valence2 = track2.get('valence', 0)

        valence_diff = abs(valence1 - valence2)

        return {
            'valence1': valence1,
            'valence2': valence2,
            'difference': valence_diff,
            'compatible': valence_diff < 0.4  # Similar mood
        }

    def _get_compatibility_recommendation(self, similarity: float) -> str:
        """Get compatibility recommendation based on similarity score."""
        if similarity > 0.8:
            return "Excellent match - perfect for seamless transitions"
        elif similarity > 0.6:
            return "Good compatibility - works well in playlist"
        elif similarity > 0.4:
            return "Moderate compatibility - may need transition track"
        else:
            return "Low compatibility - consider different pairing"

    def _generate_analysis_summary(self, tracks: List[Dict], results: Dict) -> Dict:
        """Generate a summary of the analysis results."""
        return {
            'total_tracks': len(tracks),
            'tracks_with_previews': len([t for t in tracks if t.get('preview_url')]),
            'audio_features_extracted': results.get('audio_features_extracted', 0),
            'analysis_coverage': results.get('audio_features_extracted', 0) / len(tracks) if tracks else 0,
            'top_genres': self._extract_top_genres(tracks),
            'listening_diversity': self._calculate_listening_diversity(tracks),
            'analysis_quality': 'high' if results.get('audio_features_extracted', 0) > 20 else 'medium'
        }

    def _extract_top_genres(self, tracks: List[Dict]) -> List[str]:
        """Extract top genres from track data."""
        # This would require additional Spotify API calls to get artist genres
        # For now, return empty list - can be implemented later
        return []

    def _calculate_listening_diversity(self, tracks: List[Dict]) -> float:
        """Calculate diversity score based on track metadata."""
        # Simple diversity calculation based on unique artists
        unique_artists = set(track.get('artist', '') for track in tracks)
        return len(unique_artists) / len(tracks) if tracks else 0

    def _store_recommendations(self, user_id: str, recommendations: Dict):
        """Store recommendations in database."""
        try:
            for rec_type, recs in recommendations.items():
                if isinstance(recs, list):
                    for rec in recs:
                        self.database.save_recommendation(
                            user_id=user_id,
                            track_id=rec.get('id'),
                            similarity_score=rec.get('similarity_score', 0),
                            recommendation_method=rec.get('recommendation_method', rec_type)
                        )
                elif isinstance(recs, dict):  # For mood-based recommendations
                    for mood, mood_recs in recs.items():
                        for rec in mood_recs:
                            self.database.save_recommendation(
                                user_id=user_id,
                                track_id=rec.get('id'),
                                similarity_score=rec.get('similarity_score', 0),
                                recommendation_method=f"{rec_type}_{mood}"
                            )
        except Exception as e:
            print(f"⚠️  Error storing recommendations: {e}")


# Additional database methods for enhanced features
class AudiophileDatabase:
    """Extension to existing database class for audio intelligence features."""

    def save_enhanced_audio_features(self, features: Dict):
        """Save enhanced audio features with additional fields."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create enhanced audio features table if not exists
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS enhanced_audio_features (
                        track_id TEXT PRIMARY KEY,
                        -- Spectral features
                        mfcc_0_mean REAL, mfcc_0_std REAL,
                        mfcc_1_mean REAL, mfcc_1_std REAL,
                        mfcc_2_mean REAL, mfcc_2_std REAL,
                        mfcc_3_mean REAL, mfcc_3_std REAL,
                        mfcc_4_mean REAL, mfcc_4_std REAL,
                        mfcc_5_mean REAL, mfcc_5_std REAL,
                        mfcc_6_mean REAL, mfcc_6_std REAL,
                        mfcc_7_mean REAL, mfcc_7_std REAL,
                        mfcc_8_mean REAL, mfcc_8_std REAL,
                        mfcc_9_mean REAL, mfcc_9_std REAL,
                        mfcc_10_mean REAL, mfcc_10_std REAL,
                        mfcc_11_mean REAL, mfcc_11_std REAL,
                        mfcc_12_mean REAL, mfcc_12_std REAL,
                        spectral_centroid_mean REAL, spectral_centroid_std REAL,
                        spectral_rolloff_mean REAL,
                        spectral_bandwidth_mean REAL,
                        zero_crossing_rate_mean REAL,
                        -- Rhythmic features
                        tempo_essentia REAL, tempo_librosa REAL,
                        beats_confidence REAL,
                        num_beats INTEGER, num_beats_librosa INTEGER,
                        beat_interval_mean REAL, beat_interval_std REAL,
                        -- Tonal features
                        key TEXT, scale TEXT, key_strength REAL,
                        chroma_0_mean REAL, chroma_1_mean REAL,
                        chroma_2_mean REAL, chroma_3_mean REAL,
                        chroma_4_mean REAL, chroma_5_mean REAL,
                        chroma_6_mean REAL, chroma_7_mean REAL,
                        chroma_8_mean REAL, chroma_9_mean REAL,
                        chroma_10_mean REAL, chroma_11_mean REAL,
                        -- Energy and dynamics
                        loudness_essentia REAL,
                        dynamic_complexity REAL,
                        rms_energy_mean REAL, rms_energy_std REAL,
                        -- Metadata
                        duration_analyzed REAL,
                        sample_rate INTEGER,
                        analysis_method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (track_id) REFERENCES tracks (id)
                    )
                ''')

                # Insert features
                columns = list(features.keys())
                placeholders = ', '.join(['?' for _ in columns])
                values = [features.get(col) for col in columns]

                cursor.execute(f'''
                    INSERT OR REPLACE INTO enhanced_audio_features
                    ({', '.join(columns)}) VALUES ({placeholders})
                ''', values)

                conn.commit()

        except Exception as e:
            print(f"❌ Error saving enhanced audio features: {e}")

    def save_taste_profile(self, user_id: str, taste_profile: Dict):
        """Save user taste profile to database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Convert taste profile to JSON
                profile_json = json.dumps(taste_profile)

                cursor.execute('''
                    INSERT OR REPLACE INTO taste_profiles (
                        user_id, profile_name, profile_data, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (user_id, 'main', profile_json, datetime.now(), datetime.now()))

                conn.commit()

        except Exception as e:
            print(f"❌ Error saving taste profile: {e}")

    def save_recommendation(self, user_id: str, track_id: str,
                          similarity_score: float, recommendation_method: str):
        """Save recommendation to database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO recommendations (
                        user_id, track_id, similarity_score, recommended_by
                    ) VALUES (?, ?, ?, ?)
                ''', (user_id, track_id, similarity_score, recommendation_method))

                conn.commit()

        except Exception as e:
            print(f"❌ Error saving recommendation: {e}")
