"""
Data Collection Module

Fetches user's listening data from Spotify API including:
- Top tracks over different time periods
- Recently played tracks
- Saved tracks (liked songs)
- Audio features for all collected tracks
"""

import spotipy
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class SpotifyDataCollector:
    """Collects and processes user listening data from Spotify."""

    def __init__(self, spotify_client: spotipy.Spotify):
        """
        Initialize the data collector.

        Args:
            spotify_client: Authenticated Spotify client
        """
        self.spotify = spotify_client
        self.top_tracks_limit = int(os.getenv('TOP_TRACKS_LIMIT', 50))

    def get_top_tracks(self, time_range: str = 'medium_term', limit: Optional[int] = None) -> List[Dict]:
        """
        Get user's top tracks for a specified time period.

        Args:
            time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (all time)
            limit: Number of tracks to fetch (default from env)

        Returns:
            List of track dictionaries with basic info
        """
        if limit is None:
            limit = self.top_tracks_limit

        try:
            results = self.spotify.current_user_top_tracks(
                limit=limit,
                time_range=time_range
            )

            tracks = []
            for item in results['items']:
                track_data = self._extract_track_info(item)
                track_data['source'] = f'top_tracks_{time_range}'
                tracks.append(track_data)

            print(f"✅ Collected {len(tracks)} top tracks ({time_range})")
            return tracks

        except Exception as e:
            print(f"❌ Error fetching top tracks: {e}")
            return []

    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """
        Get user's recently played tracks.

        Args:
            limit: Number of recent tracks to fetch

        Returns:
            List of track dictionaries
        """
        try:
            results = self.spotify.current_user_recently_played(limit=limit)

            tracks = []
            seen_track_ids = set()  # Avoid duplicates

            for item in results['items']:
                track = item['track']
                if track['id'] not in seen_track_ids:
                    track_data = self._extract_track_info(track)
                    track_data['source'] = 'recently_played'
                    track_data['played_at'] = item['played_at']
                    tracks.append(track_data)
                    seen_track_ids.add(track['id'])

            print(f"✅ Collected {len(tracks)} recently played tracks")
            return tracks

        except Exception as e:
            print(f"❌ Error fetching recently played tracks: {e}")
            return []

    def get_saved_tracks(self, limit: int = 50) -> List[Dict]:
        """
        Get user's saved (liked) tracks.

        Args:
            limit: Number of saved tracks to fetch

        Returns:
            List of track dictionaries
        """
        try:
            tracks = []
            offset = 0
            batch_size = 50  # Spotify API limit

            while len(tracks) < limit:
                current_limit = min(batch_size, limit - len(tracks))
                results = self.spotify.current_user_saved_tracks(
                    limit=current_limit,
                    offset=offset
                )

                if not results['items']:
                    break

                for item in results['items']:
                    track_data = self._extract_track_info(item['track'])
                    track_data['source'] = 'saved_tracks'
                    track_data['added_at'] = item['added_at']
                    tracks.append(track_data)

                offset += current_limit

            print(f"✅ Collected {len(tracks)} saved tracks")
            return tracks

        except Exception as e:
            print(f"❌ Error fetching saved tracks: {e}")
            return []

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Get audio features for a list of tracks.

        Args:
            track_ids: List of Spotify track IDs

        Returns:
            List of audio feature dictionaries
        """
        if not track_ids:
            return []

        try:
            # Spotify API allows max 100 IDs per request
            all_features = []
            batch_size = 100

            for i in range(0, len(track_ids), batch_size):
                batch_ids = track_ids[i:i + batch_size]
                features = self.spotify.audio_features(batch_ids)

                # Filter out None values (tracks without features)
                valid_features = [f for f in features if f is not None]
                all_features.extend(valid_features)

            print(f"✅ Collected audio features for {len(all_features)}/{len(track_ids)} tracks")
            return all_features

        except Exception as e:
            print(f"❌ Error fetching audio features: {e}")
            return []

    def collect_user_profile_data(self) -> Dict[str, any]:
        """
        Collect comprehensive user profile data.

        Returns:
            Dictionary containing all user data including tracks and features
        """
        print("🎵 Starting comprehensive data collection...")

        all_tracks = []

        # Collect different types of tracks
        print("\n📊 Collecting top tracks...")
        all_tracks.extend(self.get_top_tracks('short_term'))
        all_tracks.extend(self.get_top_tracks('medium_term'))
        all_tracks.extend(self.get_top_tracks('long_term'))

        print("\n🕒 Collecting recently played tracks...")
        all_tracks.extend(self.get_recently_played())

        print("\n❤️ Collecting saved tracks...")
        all_tracks.extend(self.get_saved_tracks())

        # Remove duplicates based on track ID
        unique_tracks = {}
        for track in all_tracks:
            track_id = track['id']
            if track_id not in unique_tracks:
                unique_tracks[track_id] = track
            else:
                # Keep track of all sources
                existing_sources = unique_tracks[track_id].get('source', '')
                new_source = track.get('source', '')
                if new_source not in existing_sources:
                    unique_tracks[track_id]['source'] = f"{existing_sources}, {new_source}"

        unique_tracks_list = list(unique_tracks.values())
        print(f"\n🎯 Found {len(unique_tracks_list)} unique tracks after deduplication")

        # Get audio features for all unique tracks
        print("\n🎼 Collecting audio features...")
        track_ids = [track['id'] for track in unique_tracks_list]
        audio_features = self.get_audio_features(track_ids)

        # Create features lookup
        features_dict = {f['id']: f for f in audio_features}

        # Combine track info with audio features
        enriched_tracks = []
        for track in unique_tracks_list:
            if track['id'] in features_dict:
                track['audio_features'] = features_dict[track['id']]
                enriched_tracks.append(track)

        user_data = {
            'tracks': enriched_tracks,
            'total_tracks': len(enriched_tracks),
            'collection_date': datetime.now().isoformat(),
            'track_sources': {
                'top_tracks_short': len([t for t in all_tracks if 'top_tracks_short_term' in t.get('source', '')]),
                'top_tracks_medium': len([t for t in all_tracks if 'top_tracks_medium_term' in t.get('source', '')]),
                'top_tracks_long': len([t for t in all_tracks if 'top_tracks_long_term' in t.get('source', '')]),
                'recently_played': len([t for t in all_tracks if 'recently_played' in t.get('source', '')]),
                'saved_tracks': len([t for t in all_tracks if 'saved_tracks' in t.get('source', '')])
            }
        }

        print(f"\n✅ Data collection complete! Processed {user_data['total_tracks']} tracks with audio features")
        return user_data

    def _extract_track_info(self, track: Dict) -> Dict:
        """
        Extract relevant information from a Spotify track object.

        Args:
            track: Spotify track object

        Returns:
            Dictionary with extracted track information
        """
        return {
            'id': track['id'],
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'popularity': track['popularity'],
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'release_date': track['album']['release_date'],
            'uri': track['uri'],
            'external_url': track['external_urls']['spotify']
        }

    def save_data_to_csv(self, user_data: Dict, filename: str = None) -> str:
        """
        Save collected data to CSV file.

        Args:
            user_data: User data dictionary from collect_user_profile_data
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/user_music_data_{timestamp}.csv"

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Flatten data for CSV
        rows = []
        for track in user_data['tracks']:
            row = track.copy()

            # Flatten audio features
            if 'audio_features' in track:
                features = track['audio_features']
                for key, value in features.items():
                    if key != 'id':  # Avoid duplicate ID column
                        row[f'audio_{key}'] = value
                del row['audio_features']

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        print(f"💾 Data saved to: {filename}")
        return filename


if __name__ == "__main__":
    """Test the data collection module."""
    from src.auth.spotify_auth import get_authenticated_spotify

    try:
        print("🎵 Testing Spotify Data Collection...")

        # Get authenticated client
        spotify = get_authenticated_spotify()

        # Create collector
        collector = SpotifyDataCollector(spotify)

        # Collect sample data
        user_data = collector.collect_user_profile_data()

        # Save to CSV
        csv_file = collector.save_data_to_csv(user_data)

        print(f"\n📊 Collection Summary:")
        print(f"Total unique tracks: {user_data['total_tracks']}")
        print(f"Data sources: {user_data['track_sources']}")
        print(f"Saved to: {csv_file}")

        print("✅ Data collection test completed successfully!")

    except Exception as e:
        print(f"❌ Data collection test failed: {e}")
        print("\nMake sure you have completed Spotify authentication first.")
