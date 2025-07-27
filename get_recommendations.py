#!/usr/bin/env python3
"""
Simple Music Recommendations Generator

This script helps you get music recommendations based on your collected data.
It works with your existing CSV data and the built-in recommendation system.
"""

import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append('src')

from src.auth.spotify_auth import get_authenticated_spotify
from src.database.models import AudiophileDatabase
from src.analysis.audio_intelligence import AudioIntelligenceEngine

load_dotenv()


def load_user_data(csv_path="data/user_data_20250725_125118.csv"):
    """Load user data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} tracks from {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def get_spotify_recommendations(user_tracks, n_recommendations=20):
    """Get recommendations using Spotify's recommendation API."""
    try:
        print("\n🎵 Getting Spotify API recommendations...")
        spotify = get_authenticated_spotify()

        # Get audio features for top tracks to use as seeds
        top_track_ids = user_tracks['id'].head(5).tolist()

        # Get audio features for these tracks
        audio_features = spotify.audio_features(top_track_ids)

        # Use audio features as seeds instead of track IDs
        if audio_features and any(f for f in audio_features if f):
            # Get the first valid feature set
            features = next(f for f in audio_features if f)

            # Use audio feature values as recommendation parameters
            recommendations = spotify.recommendations(
                limit=n_recommendations,
                target_danceability=features.get('danceability', 0.5),
                target_energy=features.get('energy', 0.5),
                target_valence=features.get('valence', 0.5),
                target_tempo=features.get('tempo', 120),
                seed_genres=['pop', 'hip-hop', 'indie']  # Based on your music style
            )
        else:
            # Fallback: use genre-based recommendations
            recommendations = spotify.recommendations(
                limit=n_recommendations,
                seed_genres=['pop', 'hip-hop', 'alternative', 'indie', 'country']
            )

        rec_tracks = []
        for track in recommendations['tracks']:
            # Skip if already in user's collection
            if track['id'] not in user_tracks['id'].values:
                rec_tracks.append({
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify'],
                    'recommendation_method': 'spotify_api'
                })

        return rec_tracks

    except Exception as e:
        print(f"❌ Error getting Spotify recommendations: {e}")
        return []


def get_feature_based_recommendations(user_tracks):
    """Get recommendations based on audio features if available."""
    try:
        print("\n🎼 Checking for audio features in database...")

        # Initialize database and check for features
        database = AudiophileDatabase()

        # Check if we have any tracks with features in the database
        conn = database.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM audio_features")
        feature_count = cursor.fetchone()[0]

        if feature_count > 0:
            print(f"✅ Found {feature_count} tracks with audio features")

            # Initialize the audio intelligence engine
            spotify = get_authenticated_spotify()
            intelligence_engine = AudioIntelligenceEngine(spotify, database)

            # Get user track IDs
            user_track_ids = user_tracks['id'].head(10).tolist()

            # Generate intelligent recommendations
            recommendations = intelligence_engine.generate_intelligent_recommendations(
                user_track_ids, n_recommendations=15
            )

            return recommendations
        else:
            print("⚠️  No audio features found in database. Run the full analysis first.")
            return None

    except Exception as e:
        print(f"❌ Error getting feature-based recommendations: {e}")
        return None


def simple_content_filtering(user_tracks, n_recommendations=15):
    """Simple content-based filtering using basic track metadata."""
    try:
        print("\n📊 Generating simple content-based recommendations...")

        # Analyze user's preferences from metadata
        user_artists = user_tracks['artist'].value_counts()
        top_artists = user_artists.head(5).index.tolist()

        avg_popularity = user_tracks['popularity'].mean()

        print(f"   📈 Your average track popularity: {avg_popularity:.1f}")
        print(f"   🎤 Your top artists: {', '.join(top_artists[:3])}")

        # Get recommendations using Spotify search
        spotify = get_authenticated_spotify()
        recommendations = []

        # Search for similar artists
        for artist in top_artists[:3]:
            try:
                results = spotify.search(
                    q=f'artist:"{artist}"',
                    type='track',
                    limit=5,
                    market='US'
                )

                for track in results['tracks']['items']:
                    # Skip if already in user's collection
                    if track['id'] not in user_tracks['id'].values:
                        recommendations.append({
                            'name': track['name'],
                            'artist': ', '.join([a['name'] for a in track['artists']]),
                            'album': track['album']['name'],
                            'popularity': track['popularity'],
                            'uri': track['uri'],
                            'external_url': track['external_urls']['spotify'],
                            'recommendation_method': 'content_filtering',
                            'reason': f'Similar to {artist}'
                        })
            except Exception as e:
                print(f"   ⚠️  Error searching for {artist}: {e}")
                continue

        # Remove duplicates
        seen_tracks = set()
        unique_recs = []
        for rec in recommendations:
            if rec['name'] not in seen_tracks:
                unique_recs.append(rec)
                seen_tracks.add(rec['name'])

        return unique_recs[:n_recommendations]

    except Exception as e:
        print(f"❌ Error in simple content filtering: {e}")
        return []


def display_recommendations(recommendations, title="🎯 Recommendations"):
    """Display recommendations in a nice format."""
    if not recommendations:
        print("   No recommendations found.")
        return

    print(f"\n{title}")
    print("=" * 60)

    for i, rec in enumerate(recommendations[:10], 1):
        name = rec.get('name', 'Unknown')
        artist = rec.get('artist', 'Unknown')
        method = rec.get('recommendation_method', 'Unknown')
        reason = rec.get('reason', '')
        popularity = rec.get('popularity', 0)

        print(f"{i:2d}. {name}")
        print(f"    🎤 {artist}")
        print(f"    📈 Popularity: {popularity}")
        if reason:
            print(f"    💡 {reason}")
        print(f"    🔗 {rec.get('external_url', 'No URL')}")
        print()


def save_recommendations(recommendations, filename="data/recommendations.csv"):
    """Save recommendations to CSV file."""
    try:
        if recommendations:
            df = pd.DataFrame(recommendations)
            df.to_csv(filename, index=False)
            print(f"💾 Recommendations saved to: {filename}")
        else:
            print("⚠️  No recommendations to save")
    except Exception as e:
        print(f"❌ Error saving recommendations: {e}")


def main():
    """Main function to generate recommendations."""
    print("🎵 Music Recommendation Generator")
    print("=" * 50)

    # Load user data
    user_data = load_user_data()
    if user_data is None:
        return

    print(f"\n📊 Your Music Collection:")
    print(f"   🎵 Total tracks: {len(user_data)}")
    print(f"   🎤 Unique artists: {user_data['artist'].nunique()}")
    print(f"   📀 Unique albums: {user_data['album'].nunique()}")
    print(f"   📈 Average popularity: {user_data['popularity'].mean():.1f}")

    # Show recommendation options
    print(f"\n🎯 Recommendation Options:")
    print("   1. Spotify API recommendations (quick)")
    print("   2. Advanced audio feature analysis (if available)")
    print("   3. Simple content-based filtering")
    print("   4. All methods")

    try:
        choice = input("\nSelect option (1-4): ").strip()

        all_recommendations = []

        if choice == '1' or choice == '4':
            spotify_recs = get_spotify_recommendations(user_data)
            if spotify_recs:
                display_recommendations(spotify_recs, "🎵 Spotify API Recommendations")
                all_recommendations.extend(spotify_recs)

        if choice == '2' or choice == '4':
            feature_recs = get_feature_based_recommendations(user_data)
            if feature_recs:
                # Process the complex recommendation structure
                for rec_type, recs in feature_recs.items():
                    if isinstance(recs, list) and recs:
                        display_recommendations(recs, f"🎼 {rec_type.replace('_', ' ').title()}")
                        all_recommendations.extend(recs)

        if choice == '3' or choice == '4':
            content_recs = simple_content_filtering(user_data)
            if content_recs:
                display_recommendations(content_recs, "📊 Content-Based Recommendations")
                all_recommendations.extend(content_recs)

        # Save all recommendations
        if all_recommendations:
            save_choice = input("\n💾 Save recommendations to file? (y/n): ").lower()
            if save_choice == 'y':
                save_recommendations(all_recommendations)

        print("\n✅ Recommendation generation complete!")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
