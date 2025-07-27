#!/usr/bin/env python3
"""
Smart Music Discovery & Playlist Creator

Works around API limitations to discover new music and create actual Spotify playlists.
Focuses on finding NEW artists similar to your taste.
"""

import sys
import pandas as pd
from datetime import datetime
import random
from dotenv import load_dotenv

sys.path.append('src')
from src.auth.spotify_auth import get_authenticated_spotify

load_dotenv()


def discover_similar_artists(spotify, user_data, max_searches=10):
    """Discover new artists using smart search strategies."""
    print("🔍 Discovering new artists using smart search...")

    user_artists = set(user_data['artist'].str.lower())
    discovered_tracks = []
    discovered_artists = set()
    user_track_ids = set(user_data['id'])

    # Strategy 1: Search for artists in similar genres/styles
    search_terms = [
        # Based on your collection - alternative/indie pop
        "indie pop female artist",
        "alternative pop singer",
        "emo pop artist",
        "indie rock band",
        "alternative hip hop artist",

        # Mood-based searches
        "sad indie artist",
        "alternative rap",
        "indie country artist",
        "pop punk singer",
        "emotional indie music",

        # Year-based to find fresh artists
        "indie artist 2024",
        "new alternative artist",
        "emerging pop artist",
    ]

    for search_term in search_terms[:max_searches]:
        try:
            print(f"   🔎 Searching: {search_term}")

            # Search for artists
            results = spotify.search(q=search_term, type='artist', limit=5)

            for artist in results['artists']['items']:
                artist_name = artist['name']

                # Skip if we already know this artist
                if artist_name.lower() in user_artists or artist_name in discovered_artists:
                    continue

                # Skip very unpopular artists (likely not good quality)
                if artist['popularity'] < 20:
                    continue

                discovered_artists.add(artist_name)
                print(f"      ✨ Found new artist: {artist_name} (Pop: {artist['popularity']})")

                # Get their top tracks
                try:
                    top_tracks = spotify.artist_top_tracks(artist['id'], country='US')

                    added_count = 0
                    for track in top_tracks['tracks'][:3]:  # Max 3 tracks per artist
                        if track['id'] not in user_track_ids:
                            discovered_tracks.append({
                                'name': track['name'],
                                'artist': artist_name,
                                'album': track['album']['name'],
                                'popularity': track['popularity'],
                                'uri': track['uri'],
                                'external_url': track['external_urls']['spotify'],
                                'preview_url': track.get('preview_url'),
                                'discovery_method': f'search_{search_term.replace(" ", "_")}',
                                'artist_popularity': artist['popularity']
                            })
                            added_count += 1

                except Exception as e:
                    print(f"        ⚠️  Error getting tracks for {artist_name}: {e}")
                    continue

                # Don't overwhelm with too many artists from one search
                if len(discovered_artists) >= 15:
                    break

        except Exception as e:
            print(f"   ⚠️  Search error for '{search_term}': {e}")
            continue

    print(f"✅ Found {len(discovered_tracks)} tracks from {len(discovered_artists)} new artists")
    return discovered_tracks


def discover_by_genre_combinations(spotify, user_data):
    """Discover music by combining genres that match user's taste."""
    print("\n🎵 Discovering by genre combinations...")

    user_track_ids = set(user_data['id'])
    user_artists = set(user_data['artist'].str.lower())
    discovered_tracks = []

    # Genre combinations based on your music style
    genre_combos = [
        ['indie', 'pop', 'alternative'],
        ['alternative', 'hip-hop', 'indie'],
        ['pop', 'rock', 'indie'],
        ['country', 'alternative', 'indie'],
        ['indie', 'folk', 'alternative'],
        ['electronic', 'indie', 'pop'],
        ['alternative', 'emo', 'rock'],
        ['indie', 'singer-songwriter', 'pop']
    ]

    for genres in genre_combos[:4]:  # Limit to avoid too many API calls
        try:
            print(f"   🎯 Trying genres: {', '.join(genres)}")

            # Use Spotify recommendations with genre seeds
            recommendations = spotify.recommendations(
                seed_genres=genres,
                limit=10,
                target_popularity=50,  # Medium popularity for discovery
                min_popularity=25,     # Avoid very unknown tracks
                max_popularity=85      # Avoid overly mainstream
            )

            for track in recommendations['tracks']:
                track_artist = ', '.join([a['name'] for a in track['artists']])

                # Skip if user already knows this track/artist
                if (track['id'] in user_track_ids or
                    track_artist.lower() in user_artists):
                    continue

                discovered_tracks.append({
                    'name': track['name'],
                    'artist': track_artist,
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'discovery_method': f'genre_combo_{"+".join(genres)}',
                    'genres': ', '.join(genres)
                })

        except Exception as e:
            print(f"   ⚠️  Genre combo error: {e}")
            continue

    print(f"✅ Found {len(discovered_tracks)} tracks through genre combinations")
    return discovered_tracks


def discover_by_similar_track_features(spotify, user_data):
    """Find tracks with similar characteristics to user's favorites."""
    print("\n🎼 Discovering tracks with similar musical characteristics...")

    user_track_ids = set(user_data['id'])
    user_artists = set(user_data['artist'].str.lower())
    discovered_tracks = []

    # Use your track characteristics for targeted discovery
    # Based on your data: medium-high popularity, mix of energy levels
    search_params = [
        {
            'target_energy': 0.7,
            'target_valence': 0.5,
            'target_danceability': 0.6,
            'description': 'energetic_tracks'
        },
        {
            'target_energy': 0.4,
            'target_valence': 0.3,
            'target_danceability': 0.4,
            'description': 'chill_emotional_tracks'
        },
        {
            'target_energy': 0.6,
            'target_valence': 0.7,
            'target_danceability': 0.7,
            'description': 'upbeat_pop_tracks'
        }
    ]

    for params in search_params:
        try:
            description = params.pop('description')
            print(f"   🎯 Finding {description.replace('_', ' ')}")

            recommendations = spotify.recommendations(
                seed_genres=['indie', 'alternative', 'pop'],
                limit=8,
                target_popularity=55,
                min_popularity=30,
                max_popularity=80,
                **params
            )

            for track in recommendations['tracks']:
                track_artist = ', '.join([a['name'] for a in track['artists']])

                # Skip if user already knows this track/artist
                if (track['id'] in user_track_ids or
                    track_artist.lower() in user_artists):
                    continue

                discovered_tracks.append({
                    'name': track['name'],
                    'artist': track_artist,
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'uri': track['uri'],
                    'external_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'discovery_method': f'audio_features_{description}',
                    'feature_profile': description
                })

        except Exception as e:
            print(f"   ⚠️  Feature-based discovery error: {e}")
            continue

    print(f"✅ Found {len(discovered_tracks)} tracks matching your musical DNA")
    return discovered_tracks


def create_discovery_playlist(spotify, tracks, playlist_name=None):
    """Create an actual Spotify playlist with discovered tracks."""
    try:
        if not tracks:
            print("❌ No tracks to create playlist with")
            return None

        user_info = spotify.current_user()
        user_id = user_info['id']

        # Auto-generate playlist name
        if not playlist_name:
            timestamp = datetime.now().strftime("%B %d")
            playlist_name = f"🎵 Discovery Playlist - {timestamp}"

        # Create description
        new_artists = len(set(track['artist'] for track in tracks))
        description = f"🎯 {len(tracks)} fresh tracks from {new_artists} new artists. Curated based on your taste by Audiophile AI. Discover your next favorite song!"

        print(f"\n🎵 Creating playlist: '{playlist_name}'")
        print(f"📝 With {len(tracks)} tracks from {new_artists} new artists")

        # Create playlist
        playlist = spotify.user_playlist_create(
            user=user_id,
            name=playlist_name,
            description=description,
            public=False
        )

        # Add tracks
        track_uris = [track['uri'] for track in tracks]

        # Add in batches
        batch_size = 100
        for i in range(0, len(track_uris), batch_size):
            batch = track_uris[i:i + batch_size]
            spotify.playlist_add_items(playlist['id'], batch)

        playlist_url = playlist['external_urls']['spotify']

        print(f"✅ Playlist created successfully!")
        print(f"🔗 {playlist_url}")

        return {
            'id': playlist['id'],
            'url': playlist_url,
            'name': playlist_name,
            'track_count': len(tracks)
        }

    except Exception as e:
        print(f"❌ Error creating playlist: {e}")
        return None


def display_discoveries(tracks):
    """Display discovered tracks in a nice format."""
    if not tracks:
        print("❌ No discoveries to display")
        return

    print(f"\n🎯 Discovery Summary: {len(tracks)} New Tracks")
    print("=" * 60)

    # Group by artist
    by_artist = {}
    for track in tracks:
        artist = track['artist']
        if artist not in by_artist:
            by_artist[artist] = []
        by_artist[artist].append(track)

    print(f"🎤 {len(by_artist)} New Artists Discovered:")

    for i, (artist, artist_tracks) in enumerate(by_artist.items(), 1):
        print(f"\n{i:2d}. {artist} ({len(artist_tracks)} tracks)")
        for track in artist_tracks[:2]:  # Show max 2 tracks per artist
            print(f"    🎵 {track['name']} (Pop: {track['popularity']})")
            print(f"    🔗 {track['external_url']}")


def main():
    """Main discovery and playlist creation."""
    print("🎵 Smart Music Discovery & Playlist Creator")
    print("=" * 55)

    # Load user data
    try:
        user_data = pd.read_csv("data/user_data_20250725_125118.csv")
        print(f"✅ Loaded {len(user_data)} tracks from your collection")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    try:
        spotify = get_authenticated_spotify()

        print(f"\n📊 Your Collection:")
        print(f"   🎵 {len(user_data)} tracks")
        print(f"   🎤 {user_data['artist'].nunique()} artists")
        print(f"   📈 Avg popularity: {user_data['popularity'].mean():.1f}")

        # Discover new music using multiple strategies
        all_discoveries = []

        # Strategy 1: Smart artist search
        search_discoveries = discover_similar_artists(spotify, user_data, max_searches=8)
        all_discoveries.extend(search_discoveries)

        # Strategy 2: Genre combinations
        genre_discoveries = discover_by_genre_combinations(spotify, user_data)
        all_discoveries.extend(genre_discoveries)

        # Strategy 3: Audio feature matching
        feature_discoveries = discover_by_similar_track_features(spotify, user_data)
        all_discoveries.extend(feature_discoveries)

        # Remove duplicates
        seen = set()
        unique_discoveries = []
        for track in all_discoveries:
            key = f"{track['name']}_{track['artist']}"
            if key not in seen:
                unique_discoveries.append(track)
                seen.add(key)

        # Sort by popularity (mix of popular and underground)
        unique_discoveries.sort(key=lambda x: x['popularity'], reverse=True)

        # Create a balanced mix: some popular, some underground
        popular_tracks = [t for t in unique_discoveries if t['popularity'] >= 50]
        underground_tracks = [t for t in unique_discoveries if t['popularity'] < 50]

        # Mix them for a balanced playlist
        balanced_playlist = []
        balanced_playlist.extend(popular_tracks[:15])  # 15 popular tracks
        balanced_playlist.extend(underground_tracks[:10])  # 10 underground gems

        # Shuffle for variety
        random.shuffle(balanced_playlist)

        # Limit to reasonable size
        final_tracks = balanced_playlist[:25]

        if final_tracks:
            display_discoveries(final_tracks)

            # Create playlist
            create = input(f"\n🎵 Create Spotify playlist with {len(final_tracks)} new tracks? (y/n): ").lower()

            if create == 'y':
                custom_name = input("Playlist name (Enter for auto-generated): ").strip()
                name = custom_name if custom_name else None

                playlist_info = create_discovery_playlist(spotify, final_tracks, name)

                if playlist_info:
                    print(f"\n🎉 SUCCESS! Your discovery playlist is ready!")
                    print(f"🎧 Open in Spotify: {playlist_info['url']}")
                    print(f"💡 Start listening and discover your next favorite artists!")

                    # Save for reference
                    df = pd.DataFrame(final_tracks)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    df.to_csv(f"data/discovery_{timestamp}.csv", index=False)
                    print(f"💾 Discovery saved to data/discovery_{timestamp}.csv")
        else:
            print("\n❌ No new music discovered. Your taste is very unique!")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
