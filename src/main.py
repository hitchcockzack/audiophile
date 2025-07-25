"""
Audiophile - Main Application

Orchestrates the entire workflow:
1. Authenticate with Spotify
2. Collect user listening data
3. Analyze audio features and build taste profile
4. Generate recommendations
5. Create/update playlist
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.auth.spotify_auth import get_authenticated_spotify
from src.data.collector import SpotifyDataCollector

load_dotenv()


def main():
    """Main application workflow."""
    print("🎵 Welcome to Audiophile - Your Intelligent Music Discovery Engine!")
    print("=" * 60)

    try:
        # Step 1: Authentication
        print("\n🔐 Step 1: Authenticating with Spotify...")
        spotify = get_authenticated_spotify()

        # Step 2: Data Collection
        print("\n📊 Step 2: Collecting your music data...")
        collector = SpotifyDataCollector(spotify)
        user_data = collector.collect_user_profile_data()

        # Save collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = collector.save_data_to_csv(user_data, f"data/user_data_{timestamp}.csv")

        print(f"\n✅ Successfully collected data for {user_data['total_tracks']} tracks!")
        print(f"💾 Data saved to: {csv_file}")

        # Step 3: Audio Analysis (coming next)
        print("\n🎼 Step 3: Audio analysis coming soon...")
        print("   - Will analyze audio features")
        print("   - Build your unique taste profile")
        print("   - Find patterns in your music preferences")

        # Step 4: Recommendations (coming next)
        print("\n🔍 Step 4: Recommendation engine coming soon...")
        print("   - Find new tracks matching your sound profile")
        print("   - Filter by release date and popularity")
        print("   - Score matches based on audio similarity")

        # Step 5: Playlist Creation (coming next)
        print("\n🎵 Step 5: Playlist creation coming soon...")
        print("   - Create 'Sound Mirror' playlist")
        print("   - Add recommended tracks")
        print("   - Update daily with fresh discoveries")

        print("\n" + "=" * 60)
        print("🎯 MVP Phase 1 Complete!")
        print("✅ Spotify integration working")
        print("✅ Data collection functional")
        print("📋 Next: Implement audio analysis and recommendations")

    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have set up your .env file with Spotify credentials")
        print("2. Check that your Spotify app configuration is correct")
        print("3. Make sure you have installed all requirements: pip install -r requirements.txt")
        return 1

    return 0


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
    print("🎵 Audiophile - Intelligent Music Discovery")
    print("Starting application...")

    if not setup_environment():
        sys.exit(1)

    exit_code = main()
    sys.exit(exit_code)
