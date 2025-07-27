#!/usr/bin/env python3
"""
Music Discovery Playlist Creator

Creates actual Spotify playlists with NEW music you've never heard before.
Finds similar artists and discovers fresh tracks based on your taste.
"""

import sys
import pandas as pd
from datetime import datetime
import random
from dotenv import load_dotenv

sys.path.append('src')
from src.auth.spotify_auth import get_authenticated_spotify

load_dotenv()


def get_related_artists(spotify, artist_id, max_artists=5):
    """Get artists similar to a given artist."""
    try:
        related = spotify.artist_related_artists(artist_id)
        return related['artists'][:max_artists]
    except Exception as e:
        print(f"   ⚠️  Error getting related artists: {e}")
        return []


def find_artist_id(spotify, artist_name):
    """Find Spotify artist ID by name."""
    try:
        results = spotify.search(q=f'artist:"{artist_name}"', type='artist', limit=1)
        if results['artists']['items']:
            return results['artists']['items'][0]['id']
        return None
    except Exception as e:
        print(f"   ⚠️  Error finding artist {artist_name}: {e}")
        return None


def get_artist_top_tracks(spotify, artist_id, market='US'):
    """Get an artist's top tracks."""
    try:
        top_tracks = spotify.artist_top_tracks(artist_id, country=market)
        return top_tracks['tracks']
    except Exception as e:
        print(f"   ⚠️  Error getting top tracks: {e}")
        return []


def discover_new_music(spotify, user_tracks, tracks_per_artist=2, max_new_artists=10):
    """Discover new music from artists similar to user's favorites."""
    print("🔍 Discovering new artists similar to your taste...")

    # Get user's top artists
    user_artists = user_tracks['artist'].value_counts()
    top_user_artists = user_artists.head(5).index.tolist()

    print(f"   🎤 Your top artists: {', '.join(top_user_artists)}")

    discovered_tracks = []
    discovered_artists = set()
    user_track_ids = set(user_tracks['id'].tolist())

    for artist_name in top_user_artists:
        print(f"\n   🔎 Finding artists similar to {artist_name}...")

        # Find the artist ID
        artist_id = find_artist_id(spotify, artist_name)
        if not artist_id:
            continue

        # Get related artists
        related_artists = get_related_artists(spotify, artist_id, max_artists=3)

        for related_artist in related_artists:
            if related_artist['name'] in discovered_artists:
                continue

            # Skip if this is an artist user already listens to
            if related_artist['name'] in user_artists.index:
                continue

            discovered_artists.add(related_artist['name'])
            print(f"      ✨ Found: {related_artist['name']} (Popularity: {related_artist['popularity']})")

            # Get some top tracks from this new artist
            top_tracks = get_artist_top_tracks(spotify, related_artist['id'])

            added_count = 0
            for track in top_tracks:
                # Skip if user already has this track
                if track['id'] in user_track_ids:
                    continue

                discovered_tracks.append({
                    'name': track['name'],
                    'artist': related_artist['name'],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'discovery_method': f'similar_to_{artist_name}',
                    'artist_popularity': related_artist['popularity']
                })

                added_count += 1
                if added_count >= tracks_per_artist:
                    break

            if len(discovered_artists) >= max_new_artists:
                break

        if len(discovered_artists) >= max_new_artists:
            break

    print(f"\n✅ Discovered {len(discovered_tracks)} tracks from {len(discovered_artists)} new artists!")
    return discovered_tracks


def get_genre_based_discoveries(spotify, user_tracks, n_tracks=10):
    """Get additional discoveries based on audio features of user's music."""
    print("\n🎵 Finding genre-based discoveries...")

    try:
        # Analyze user's track characteristics
        avg_popularity = user_tracks['popularity'].mean()

        # Get a few track IDs to analyze
        sample_track_ids = user_tracks['id'].head(10).tolist()
        audio_features = spotify.audio_features(sample_track_ids)

        # Calculate average characteristics
        valid_features = [f for f in audio_features if f]
        if valid_features:
            avg_energy = sum(f['energy'] for f in valid_features) / len(valid_features)
            avg_valence = sum(f['valence'] for f in valid_features) / len(valid_features)
            avg_danceability = sum(f['danceability'] for f in valid_features) / len(valid_features)

            print(f"   📊 Your music profile: Energy={avg_energy:.2f}, Mood={avg_valence:.2f}, Dance={avg_danceability:.2f}")

            # Use these characteristics to find similar tracks
            recommendations = spotify.recommendations(
                limit=n_tracks * 2,  # Get extra in case some are filtered out
                target_energy=avg_energy,
                target_valence=avg_valence,
                target_danceability=avg_danceability,
                target_popularity=min(max(int(avg_popularity), 30), 80),  # Keep reasonable range
                seed_genres=['alternative', 'indie', 'pop', 'hip-hop', 'rock']
            )

            user_track_ids = set(user_tracks['id'].tolist())
            user_artists = set(user_tracks['artist'].tolist())

            genre_discoveries = []
            for track in recommendations['tracks']:
                # Skip if user already has this track or artist
                track_artist = ', '.join([artist['name'] for artist in track['artists']])
                if track['id'] in user_track_ids or track_artist in user_artists:
                    continue

                genre_discoveries.append({
                    'name': track['name'],
                    'artist': track_artist,
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'discovery_method': 'audio_features_match'
                })

                if len(genre_discoveries) >= n_tracks:
                    break

            print(f"   ✅ Found {len(genre_discoveries)} tracks matching your musical DNA")
            return genre_discoveries

    except Exception as e:
        print(f"   ⚠️  Error in genre-based discovery: {e}")

    return []


def create_spotify_playlist(spotify, tracks, playlist_name=None):
    """Create an actual Spotify playlist with the discovered tracks."""
    try:
        if not tracks:
            print("❌ No tracks to add to playlist")
            return None

        # Get user info
        user_info = spotify.current_user()
        user_id = user_info['id']

        # Generate playlist name if not provided
        if not playlist_name:
            timestamp = datetime.now().strftime("%B %d")
            playlist_name = f"🎵 Music Discovery - {timestamp}"

        # Create playlist description
        artists_found = len(set(track['artist'] for track in tracks))
        description = f"🎯 {len(tracks)} fresh tracks from {artists_found} new artists based on your taste. Automatically curated by Audiophile AI."

        print(f"\n🎵 Creating Spotify playlist: '{playlist_name}'")

        # Create the playlist
        playlist = spotify.user_playlist_create(
            user=user_id,
            name=playlist_name,
            description=description,
            public=False  # Keep private by default
        )

        # Prepare track URIs
        track_uris = [track['uri'] for track in tracks if track.get('uri')]

        if track_uris:
            # Add tracks in batches (Spotify limit is 100 per request)
            batch_size = 100
            for i in range(0, len(track_uris), batch_size):
                batch = track_uris[i:i + batch_size]
                spotify.playlist_add_items(playlist['id'], batch)

            playlist_url = playlist['external_urls']['spotify']

            print(f"✅ Playlist created successfully!")
            print(f"🔗 Spotify URL: {playlist_url}")
            print(f"📊 Added {len(track_uris)} tracks to playlist")

            return {
                'playlist_id': playlist['id'],
                'playlist_url': playlist_url,
                'name': playlist_name,
                'track_count': len(track_uris)
            }
        else:
            print("❌ No valid track URIs found")
            return None

    except Exception as e:
        print(f"❌ Error creating playlist: {e}")
        return None


def display_discovery_summary(discovered_tracks):
    """Display a summary of discovered music."""
    if not discovered_tracks:
        print("❌ No new music discovered")
        return

    print(f"\n🎯 Music Discovery Summary")
    print("=" * 50)

    # Group by discovery method
    by_method = {}
    for track in discovered_tracks:
        method = track.get('discovery_method', 'unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(track)

    for method, tracks in by_method.items():
        print(f"\n🔹 {method.replace('_', ' ').title()}: {len(tracks)} tracks")

        # Show top 3 tracks from this method
        for track in tracks[:3]:
            print(f"   • {track['name']} - {track['artist']} (Pop: {track['popularity']})")

    # Show new artists discovered
    new_artists = list(set(track['artist'] for track in discovered_tracks))
    print(f"\n🎤 New Artists Discovered ({len(new_artists)}):")
    for artist in new_artists[:10]:  # Show first 10
        print(f"   • {artist}")

    if len(new_artists) > 10:
        print(f"   ... and {len(new_artists) - 10} more!")


def main():
    """Main discovery and playlist creation function."""
    print("🎵 Music Discovery & Playlist Creator")
    print("=" * 50)

    # Load user data
    try:
        user_data = pd.read_csv("data/user_data_20250725_125118.csv")
        print(f"✅ Loaded {len(user_data)} tracks from your collection")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    try:
        spotify = get_authenticated_spotify()

        print(f"\n📊 Your Music Profile:")
        print(f"   🎵 Total tracks: {len(user_data)}")
        print(f"   🎤 Unique artists: {user_data['artist'].nunique()}")
        print(f"   📈 Average popularity: {user_data['popularity'].mean():.1f}")

        # Discover new music through multiple methods
        all_discoveries = []

        # Method 1: Related artists discovery
        related_discoveries = discover_new_music(
            spotify, user_data,
            tracks_per_artist=2,
            max_new_artists=8
        )
        all_discoveries.extend(related_discoveries)

        # Method 2: Audio features matching
        genre_discoveries = get_genre_based_discoveries(spotify, user_data, n_tracks=8)
        all_discoveries.extend(genre_discoveries)

        # Remove duplicates and shuffle
        seen_tracks = set()
        unique_discoveries = []
        for track in all_discoveries:
            track_key = f"{track['name']}_{track['artist']}"
            if track_key not in seen_tracks:
                unique_discoveries.append(track)
                seen_tracks.add(track_key)

        # Shuffle for variety
        random.shuffle(unique_discoveries)

        # Limit to reasonable playlist size
        final_tracks = unique_discoveries[:25]

        # Display summary
        display_discovery_summary(final_tracks)

        if final_tracks:
            # Ask if user wants to create playlist
            create_playlist = input(f"\n🎵 Create Spotify playlist with {len(final_tracks)} new tracks? (y/n): ").lower()

            if create_playlist == 'y':
                # Custom playlist name
                custom_name = input("Enter playlist name (or press Enter for auto-generated): ").strip()
                playlist_name = custom_name if custom_name else None

                playlist_info = create_spotify_playlist(spotify, final_tracks, playlist_name)

                if playlist_info:
                    print(f"\n🎉 SUCCESS! Your discovery playlist is ready!")
                    print(f"🔗 Open in Spotify: {playlist_info['playlist_url']}")

                    # Save discoveries to file for reference
                    df = pd.DataFrame(final_tracks)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/discoveries_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                    print(f"💾 Discovery data saved to: {filename}")
        else:
            print("\n❌ No new music discovered. Try adjusting the discovery parameters.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
