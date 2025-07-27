#!/usr/bin/env python3
"""
Quick Music Recommendations - No API Issues

Simple genre and artist-based recommendations that work reliably.
"""

import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append('src')
from src.auth.spotify_auth import get_authenticated_spotify

load_dotenv()


def get_genre_recommendations(spotify, genres, n_recommendations=20):
    """Get recommendations based on genres."""
    try:
        print(f"🎵 Getting recommendations for genres: {', '.join(genres)}")

        recommendations = spotify.recommendations(
            limit=n_recommendations,
            seed_genres=genres
        )

        rec_tracks = []
        for track in recommendations['tracks']:
            rec_tracks.append({
                'name': track['name'],
                'artist': ', '.join([artist['name'] for artist in track['artists']]),
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'uri': track['uri'],
                'external_url': track['external_urls']['spotify'],
                'genres': ', '.join(genres)
            })

        return rec_tracks

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def main():
    """Get quick genre-based recommendations."""
    print("🎵 Quick Music Recommendations")
    print("=" * 40)

    # Based on your music collection analysis
    user_genres = {
        'Alternative/Indie': ['alternative', 'indie', 'indie-pop'],
        'Hip-Hop/Rap': ['hip-hop', 'rap', 'trap'],
        'Pop': ['pop', 'pop-rock', 'electropop'],
        'Country': ['country', 'country-rock'],
        'Rock/Metal': ['rock', 'metal', 'alternative-rock']
    }

    print("\n🎯 Choose your mood/genre:")
    print("1. Alternative/Indie vibes")
    print("2. Hip-Hop/Rap energy")
    print("3. Pop hits")
    print("4. Country feels")
    print("5. Rock/Metal power")
    print("6. Mix of everything")

    try:
        spotify = get_authenticated_spotify()
        choice = input("\nSelect (1-6): ").strip()

        if choice == '1':
            recs = get_genre_recommendations(spotify, user_genres['Alternative/Indie'])
        elif choice == '2':
            recs = get_genre_recommendations(spotify, user_genres['Hip-Hop/Rap'])
        elif choice == '3':
            recs = get_genre_recommendations(spotify, user_genres['Pop'])
        elif choice == '4':
            recs = get_genre_recommendations(spotify, user_genres['Country'])
        elif choice == '5':
            recs = get_genre_recommendations(spotify, user_genres['Rock/Metal'])
        elif choice == '6':
            # Mix of all genres
            all_genres = ['pop', 'hip-hop', 'alternative', 'indie', 'country']
            recs = get_genre_recommendations(spotify, all_genres)
        else:
            print("Invalid choice, using mix of everything")
            all_genres = ['pop', 'hip-hop', 'alternative', 'indie', 'country']
            recs = get_genre_recommendations(spotify, all_genres)

        if recs:
            print(f"\n🎵 Found {len(recs)} recommendations:")
            print("=" * 50)

            for i, rec in enumerate(recs[:10], 1):
                print(f"{i:2d}. {rec['name']}")
                print(f"    🎤 {rec['artist']}")
                print(f"    📈 Popularity: {rec['popularity']}")
                print(f"    🔗 {rec['external_url']}")
                print()

            # Save to file
            df = pd.DataFrame(recs)
            df.to_csv('data/quick_recommendations.csv', index=False)
            print(f"💾 Saved {len(recs)} recommendations to data/quick_recommendations.csv")
        else:
            print("❌ No recommendations found")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
