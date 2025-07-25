"""
Spotify Authentication Module

Handles OAuth2 authentication flow with Spotify API using spotipy library.
Manages token caching and refresh for seamless API access.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()


class SpotifyAuthenticator:
    """Handles Spotify API authentication and token management."""

    def __init__(self):
        """Initialize the authenticator with credentials from environment variables."""
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')

        # Required scopes for the application
        self.scope = [
            'user-top-read',           # Read user's top tracks
            'user-library-read',       # Read saved tracks
            'playlist-modify-public',  # Create and modify public playlists
            'playlist-modify-private', # Create and modify private playlists
            'user-read-recently-played'# Read recently played tracks
        ]

        self._validate_credentials()
        self._setup_auth_manager()

    def _validate_credentials(self) -> None:
        """Validate that all required credentials are provided."""
        if not all([self.client_id, self.client_secret]):
            raise ValueError(
                "Missing Spotify credentials. Please check your .env file:\n"
                "- SPOTIFY_CLIENT_ID\n"
                "- SPOTIFY_CLIENT_SECRET\n"
                "Get these from: https://developer.spotify.com/dashboard"
            )

    def _setup_auth_manager(self) -> None:
        """Set up the Spotify OAuth manager."""
        self.auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=' '.join(self.scope),
            cache_path='.spotify_cache',  # Cache tokens locally
            show_dialog=True  # Force login dialog (useful for development)
        )

    def get_spotify_client(self) -> spotipy.Spotify:
        """
        Get an authenticated Spotify client.

        Returns:
            spotipy.Spotify: Authenticated Spotify client

        Raises:
            Exception: If authentication fails
        """
        try:
            spotify = spotipy.Spotify(auth_manager=self.auth_manager)

            # Test the connection by getting current user info
            user_info = spotify.current_user()
            print(f"✅ Successfully authenticated as: {user_info['display_name']} ({user_info['id']})")

            return spotify

        except Exception as e:
            raise Exception(f"Failed to authenticate with Spotify: {str(e)}")

    def get_user_info(self, spotify_client: spotipy.Spotify) -> dict:
        """
        Get current user information.

        Args:
            spotify_client: Authenticated Spotify client

        Returns:
            dict: User information including ID, display name, etc.
        """
        return spotify_client.current_user()

    def clear_cache(self) -> None:
        """Clear the cached authentication tokens."""
        cache_path = '.spotify_cache'
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("🗑️  Cleared Spotify authentication cache")


def get_authenticated_spotify() -> spotipy.Spotify:
    """
    Convenience function to get an authenticated Spotify client.

    Returns:
        spotipy.Spotify: Authenticated Spotify client
    """
    auth = SpotifyAuthenticator()
    return auth.get_spotify_client()


if __name__ == "__main__":
    """Test the authentication module."""
    try:
        print("🎵 Testing Spotify Authentication...")

        auth = SpotifyAuthenticator()
        spotify = auth.get_spotify_client()
        user_info = auth.get_user_info(spotify)

        print(f"User ID: {user_info['id']}")
        print(f"Display Name: {user_info['display_name']}")
        print(f"Followers: {user_info['followers']['total']}")
        print(f"Country: {user_info['country']}")

        print("✅ Authentication test completed successfully!")

    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have created a .env file with your Spotify credentials")
        print("2. Check that your Spotify app has the correct redirect URI")
        print("3. Ensure your Spotify app is not in development mode restrictions")
