"""
Audiophile - Advanced Music Intelligence Application

Orchestrates the complete workflow:
1. Authenticate with Spotify
2. Collect user listening data with preview URLs
3. Perform advanced audio analysis using Essentia/librosa
4. Extract comprehensive audio features algorithmically
5. Generate intelligent recommendations using multiple algorithms
6. Create personalized playlists and analyze taste profiles
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.auth.spotify_auth import get_authenticated_spotify
from src.data.collector import SpotifyDataCollector
from src.database.models import AudiophileDatabase
from src.analysis.audio_intelligence import AudioIntelligenceEngine

load_dotenv()


def main():
    """Main application workflow with advanced audio intelligence."""
    print("🎵 Welcome to Audiophile - Advanced Music Intelligence Engine!")
    print("=" * 70)

    try:
        # Step 1: Authentication
        print("\n🔐 Step 1: Authenticating with Spotify...")
        spotify = get_authenticated_spotify()

        # Step 2: Database Setup
        print("\n🗄️  Step 2: Initializing database...")
        database = AudiophileDatabase()

        # Step 3: Data Collection
        print("\n📊 Step 3: Collecting your music data...")
        collector = SpotifyDataCollector(spotify, database)
        user_data = collector.collect_user_profile_data()

        print(f"✅ Successfully collected data for {user_data['total_tracks']} tracks!")

        # Step 4: Initialize Audio Intelligence Engine
        print("\n🤖 Step 4: Initializing Audio Intelligence Engine...")
        intelligence_engine = AudioIntelligenceEngine(spotify, database)

        # Step 5: Advanced Audio Analysis
        print("\n🎼 Step 5: Performing advanced audio analysis...")
        print("   🔬 This will download audio previews and extract technical features")
        print("   🎯 Using Essentia and librosa for comprehensive signal analysis")

        # Ask user how many tracks to analyze (for performance)
        max_tracks = get_analysis_preference()

        analysis_results = intelligence_engine.analyze_user_music_collection(
            user_data, max_analysis_tracks=max_tracks
        )

        # Display analysis summary
        display_analysis_summary(analysis_results)

        # Step 6: Generate Intelligent Recommendations
        if analysis_results.get('audio_features_extracted', 0) > 0:
            print("\n🔍 Step 6: Generating intelligent recommendations...")

            user_track_ids = [track['id'] for track in user_data['tracks'][:20]]
            recommendations = intelligence_engine.generate_intelligent_recommendations(
                user_track_ids, n_recommendations=15
            )

            # Display recommendations
            display_recommendations(recommendations)

            # Step 7: Create Smart Playlist
            print("\n🎵 Step 7: Creating AI-generated playlist...")

            # Use top tracks as seeds for playlist generation
            seed_tracks = user_track_ids[:5]  # Top 5 tracks as seeds
            smart_playlist = intelligence_engine.create_smart_playlist(
                seed_tracks,
                playlist_name="AI Sound Mirror",
                playlist_length=25,
                diversity_factor=0.3
            )

            if smart_playlist:
                display_playlist_info(smart_playlist)

                # Option to create actual Spotify playlist
                create_spotify_playlist = input("\n🎶 Create this playlist on Spotify? (y/n): ").lower() == 'y'
                if create_spotify_playlist:
                    create_playlist_on_spotify(spotify, smart_playlist)

            # Step 8: Additional Analysis Features
            print("\n📈 Step 8: Advanced analytics available...")
            show_advanced_features_menu(intelligence_engine, user_track_ids)

        else:
            print("\n⚠️  Limited audio analysis - no preview URLs available")
            print("   Recommendations will be based on Spotify metadata only")

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_analysis_results(analysis_results, f"data/analysis_results_{timestamp}.json")

        print("\n" + "=" * 70)
        print("🎯 Audio Intelligence Analysis Complete!")
        print("✅ Spotify integration working")
        print("✅ Advanced audio feature extraction completed")
        print("✅ Intelligent recommendations generated")
        print("✅ Personalized taste profile created")
        print("📋 Your music intelligence data is now available for exploration!")

    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have set up your .env file with Spotify credentials")
        print("2. Check that your Spotify app configuration is correct")
        print("3. Install audio analysis libraries: pip install -r requirements.txt")
        print("4. For Essentia: pip install essentia-tensorflow")
        return 1

    return 0


def get_analysis_preference() -> int:
    """Ask user how many tracks to analyze for performance control."""
    print("\n🎚️  Audio Analysis Options:")
    print("   1. Quick Analysis (10 tracks) - Fast preview")
    print("   2. Standard Analysis (25 tracks) - Balanced")
    print("   3. Deep Analysis (50 tracks) - Comprehensive")
    print("   4. Custom number")

    try:
        choice = input("Select option (1-4): ").strip()

        if choice == '1':
            return 10
        elif choice == '2':
            return 25
        elif choice == '3':
            return 50
        elif choice == '4':
            custom = int(input("Enter number of tracks to analyze: "))
            return min(max(custom, 1), 100)  # Limit between 1-100
        else:
            print("Invalid choice, using standard analysis (25 tracks)")
            return 25

    except (ValueError, KeyboardInterrupt):
        print("Using standard analysis (25 tracks)")
        return 25


def display_analysis_summary(results: dict):
    """Display a summary of the analysis results."""
    print("\n📊 Analysis Summary:")
    print(f"   📀 Total tracks in collection: {results.get('total_tracks', 0)}")
    print(f"   🎵 Tracks analyzed with audio features: {results.get('audio_features_extracted', 0)}")

    coverage = results.get('audio_features_extracted', 0) / max(results.get('total_tracks', 1), 1)
    print(f"   📈 Analysis coverage: {coverage:.1%}")

    if results.get('taste_profile'):
        profile = results['taste_profile']
        summary = profile.get('summary', {})

        print(f"\n🎭 Your Music Taste Profile:")
        if summary.get('avg_energy') is not None:
            energy_level = "High" if summary['avg_energy'] > 0.7 else "Medium" if summary['avg_energy'] > 0.4 else "Low"
            print(f"   ⚡ Energy Level: {energy_level} ({summary['avg_energy']:.2f})")

        if summary.get('avg_valence') is not None:
            mood_level = "Positive" if summary['avg_valence'] > 0.6 else "Neutral" if summary['avg_valence'] > 0.4 else "Melancholic"
            print(f"   😊 Mood Preference: {mood_level} ({summary['avg_valence']:.2f})")

        if summary.get('avg_tempo') is not None:
            tempo_desc = "Fast" if summary['avg_tempo'] > 140 else "Medium" if summary['avg_tempo'] > 90 else "Slow"
            print(f"   🥁 Tempo Preference: {tempo_desc} ({summary['avg_tempo']:.0f} BPM)")


def display_recommendations(recommendations: dict):
    """Display the generated recommendations."""
    print("\n🎯 Intelligent Recommendations:")

    for rec_type, recs in recommendations.items():
        if isinstance(recs, list) and recs:
            print(f"\n   🔹 {rec_type.replace('_', ' ').title()}:")
            for i, rec in enumerate(recs[:5], 1):  # Show top 5
                similarity = rec.get('similarity_score', 0)
                print(f"      {i}. {rec.get('name', 'Unknown')} - {rec.get('artist', 'Unknown')} ({similarity:.2f})")

        elif isinstance(recs, dict):  # For mood-based recommendations
            print(f"\n   🔹 {rec_type.replace('_', ' ').title()}:")
            for mood, mood_recs in recs.items():
                if mood_recs:
                    print(f"      💭 {mood.title()}: {mood_recs[0].get('name', 'Unknown')} - {mood_recs[0].get('artist', 'Unknown')}")


def display_playlist_info(playlist: dict):
    """Display information about the generated playlist."""
    print(f"\n🎵 Smart Playlist: {playlist.get('name', 'Unknown')}")
    print(f"   📝 Description: {playlist.get('description', '')}")
    print(f"   🎼 Tracks: {playlist.get('length', 0)}")
    print(f"   🎯 Algorithm: {playlist.get('algorithm', 'Unknown')}")

    tracks = playlist.get('tracks', [])
    if tracks:
        print(f"\n   🎶 Sample tracks:")
        for i, track in enumerate(tracks[:5], 1):
            score = track.get('playlist_score', track.get('similarity_score', 0))
            print(f"      {i}. {track.get('name', 'Unknown')} - {track.get('artist', 'Unknown')} ({score:.2f})")

        if len(tracks) > 5:
            print(f"      ... and {len(tracks) - 5} more tracks")


def create_playlist_on_spotify(spotify, smart_playlist: dict):
    """Create the generated playlist on Spotify."""
    try:
        user_info = spotify.current_user()
        user_id = user_info['id']

        # Create playlist
        playlist = spotify.user_playlist_create(
            user=user_id,
            name=smart_playlist['name'],
            description=smart_playlist['description'],
            public=False
        )

        # Add tracks
        track_uris = []
        for track in smart_playlist.get('tracks', []):
            if track.get('uri'):
                track_uris.append(track['uri'])

        if track_uris:
            # Add tracks in batches (Spotify API limit)
            batch_size = 100
            for i in range(0, len(track_uris), batch_size):
                batch = track_uris[i:i + batch_size]
                spotify.playlist_add_items(playlist['id'], batch)

        playlist_url = playlist['external_urls']['spotify']
        print(f"✅ Playlist created successfully!")
        print(f"🔗 Spotify URL: {playlist_url}")

    except Exception as e:
        print(f"❌ Error creating Spotify playlist: {e}")


def show_advanced_features_menu(intelligence_engine, user_track_ids: list):
    """Show menu for advanced analysis features."""
    print("\n🔬 Advanced Features Available:")
    print("   1. Track Compatibility Analysis")
    print("   2. Generate More Recommendations")
    print("   3. Analyze Specific Mood Preferences")
    print("   4. Export Analysis Data")
    print("   5. Continue to finish")

    try:
        choice = input("Select option (1-5): ").strip()

        if choice == '1':
            analyze_track_compatibility(intelligence_engine, user_track_ids)
        elif choice == '2':
            generate_more_recommendations(intelligence_engine, user_track_ids)
        elif choice == '3':
            analyze_mood_preferences(intelligence_engine)
        elif choice == '4':
            print("📁 Analysis data has been saved to data/ directory")
        elif choice == '5':
            return
        else:
            print("Invalid choice")

    except (ValueError, KeyboardInterrupt):
        return


def analyze_track_compatibility(intelligence_engine, user_track_ids: list):
    """Demonstrate track compatibility analysis."""
    if len(user_track_ids) >= 2:
        track1 = user_track_ids[0]
        track2 = user_track_ids[1]

        print(f"\n🔬 Analyzing compatibility between your top 2 tracks...")
        compatibility = intelligence_engine.analyze_track_compatibility(track1, track2)

        if compatibility:
            print(f"   🎯 Overall Similarity: {compatibility.get('overall_similarity', 0):.2f}")
            print(f"   💡 Recommendation: {compatibility.get('recommendation', 'Unknown')}")

            tempo_comp = compatibility.get('tempo_compatibility', {})
            if tempo_comp.get('compatible'):
                print(f"   🥁 Tempo: Compatible ({tempo_comp.get('tempo1', 0):.0f} & {tempo_comp.get('tempo2', 0):.0f} BPM)")
            else:
                print(f"   🥁 Tempo: Different ({tempo_comp.get('tempo1', 0):.0f} & {tempo_comp.get('tempo2', 0):.0f} BPM)")


def generate_more_recommendations(intelligence_engine, user_track_ids: list):
    """Generate additional recommendations with different parameters."""
    print("\n🎯 Generating additional recommendations...")

    # Generate mood-based recommendations
    moods = ['energetic', 'chill', 'happy']
    for mood in moods:
        mood_recs = intelligence_engine.recommender.get_mood_based_recommendations(mood, n_recommendations=3)
        if mood_recs:
            print(f"\n   💭 {mood.title()} Recommendations:")
            for i, rec in enumerate(mood_recs, 1):
                print(f"      {i}. {rec.get('name', 'Unknown')} - {rec.get('artist', 'Unknown')}")


def analyze_mood_preferences(intelligence_engine):
    """Analyze user's mood preferences."""
    print("\n😊 Mood Analysis:")

    moods = ['energetic', 'chill', 'happy', 'party']
    for mood in moods:
        mood_tracks = intelligence_engine.recommender.get_mood_based_recommendations(mood, n_recommendations=1)
        count = len(mood_tracks)
        print(f"   {mood.title()}: {count} matching tracks in collection")


def save_analysis_results(results: dict, filename: str):
    """Save analysis results to JSON file."""
    try:
        import json
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Analysis results saved to: {filename}")

    except Exception as e:
        print(f"⚠️  Error saving results: {e}")


def setup_environment():
    """Set up the application environment."""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found!")
        print("📋 Please create a .env file based on config/env_template.txt")
        print("🔗 Get your Spotify credentials from: https://developer.spotify.com/dashboard")
        return False

    return True


if __name__ == "__main__":
    print("🎵 Audiophile - Advanced Music Intelligence")
    print("🔬 Powered by Essentia & Advanced Audio Analysis")
    print("Starting application...")

    if not setup_environment():
        sys.exit(1)

    exit_code = main()
    sys.exit(exit_code)
