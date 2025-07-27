"""
Relative Popularity Analysis Module

Calculates popularity relative to an artist's own catalog rather than global metrics.
Identifies tracks that are "climbing the ranks" within an artist's discography,
considering factors like:
- Track age vs listen count ratio
- Position in artist's top tracks
- Recent momentum vs established hits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from dateutil.parser import parse as parse_date


class RelativePopularityAnalyzer:
    """Analyzes track popularity relative to artist's catalog."""

    def __init__(self, spotify_client, database):
        """
        Initialize the relative popularity analyzer.

        Args:
            spotify_client: Authenticated Spotify client
            database: Database instance
        """
        self.spotify = spotify_client
        self.database = database
        self.artist_catalogs = {}  # Cache for artist track data

    def calculate_relative_popularity_score(self, track_id: str, artist_id: str) -> Dict:
        """
        Calculate relative popularity score for a track within artist's catalog.

        Args:
            track_id: Spotify track ID
            artist_id: Spotify artist ID

        Returns:
            Dictionary with relative popularity metrics
        """
        try:
            # Get artist's complete catalog
            artist_tracks = self._get_artist_catalog(artist_id)

            if not artist_tracks:
                return {'error': 'Could not fetch artist catalog'}

            # Find the target track
            target_track = next((t for t in artist_tracks if t['id'] == track_id), None)
            if not target_track:
                return {'error': 'Track not found in artist catalog'}

            # Calculate relative metrics
            popularity_rank = self._calculate_popularity_rank(target_track, artist_tracks)
            momentum_score = self._calculate_momentum_score(target_track, artist_tracks)
            discovery_score = self._calculate_discovery_score(target_track, artist_tracks)
            relative_position = self._calculate_relative_position(target_track, artist_tracks)

            return {
                'track_id': track_id,
                'artist_id': artist_id,
                'global_popularity': target_track.get('popularity', 0),
                'artist_catalog_size': len(artist_tracks),
                'popularity_rank': popularity_rank,  # 1 = most popular in catalog
                'relative_popularity_percentile': relative_position['percentile'],
                'momentum_score': momentum_score,  # How fast it's climbing
                'discovery_score': discovery_score,  # Likelihood of being a hidden gem
                'is_rising_star': momentum_score > 0.7 and relative_position['percentile'] < 0.8,
                'is_hidden_gem': discovery_score > 0.6 and target_track.get('popularity', 0) < 30,
                'recommendation_weight': self._calculate_recommendation_weight(
                    momentum_score, discovery_score, relative_position['percentile']
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error calculating relative popularity: {e}")
            return {'error': str(e)}

    def _get_artist_catalog(self, artist_id: str, force_refresh: bool = False) -> List[Dict]:
        """
        Get complete artist catalog with caching.

        Args:
            artist_id: Spotify artist ID
            force_refresh: Force refresh of cached data

        Returns:
            List of track dictionaries
        """
        if not force_refresh and artist_id in self.artist_catalogs:
            return self.artist_catalogs[artist_id]

        try:
            all_tracks = []

            # Get albums (including singles and compilations)
            albums = self._get_artist_albums(artist_id)

            for album in albums:
                # Get tracks from each album
                album_tracks = self._get_album_tracks(album['id'])

                for track in album_tracks:
                    # Enhance track data
                    enhanced_track = {
                        'id': track['id'],
                        'name': track['name'],
                        'album_id': album['id'],
                        'album_name': album['name'],
                        'album_type': album['album_type'],
                        'release_date': album['release_date'],
                        'track_number': track['track_number'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track.get('popularity', 0),
                        'explicit': track.get('explicit', False)
                    }

                    # Calculate track age in days
                    try:
                        release_date = parse_date(album['release_date'])
                        track_age_days = (datetime.now() - release_date).days
                        enhanced_track['age_days'] = track_age_days
                    except:
                        enhanced_track['age_days'] = None

                    all_tracks.append(enhanced_track)

            # Remove duplicates (same track on multiple albums)
            unique_tracks = {}
            for track in all_tracks:
                if track['id'] not in unique_tracks:
                    unique_tracks[track['id']] = track

            final_tracks = list(unique_tracks.values())

            # Cache the results
            self.artist_catalogs[artist_id] = final_tracks

            print(f"📚 Cached {len(final_tracks)} tracks for artist")
            return final_tracks

        except Exception as e:
            print(f"❌ Error fetching artist catalog: {e}")
            return []

    def _get_artist_albums(self, artist_id: str) -> List[Dict]:
        """Get all albums for an artist."""
        try:
            albums = []
            limit = 50
            offset = 0

            while True:
                results = self.spotify.artist_albums(
                    artist_id,
                    album_type='album,single,compilation',
                    limit=limit,
                    offset=offset
                )

                if not results['items']:
                    break

                albums.extend(results['items'])

                if len(results['items']) < limit:
                    break

                offset += limit

            return albums

        except Exception as e:
            print(f"❌ Error fetching artist albums: {e}")
            return []

    def _get_album_tracks(self, album_id: str) -> List[Dict]:
        """Get all tracks from an album."""
        try:
            tracks = []
            limit = 50
            offset = 0

            while True:
                results = self.spotify.album_tracks(album_id, limit=limit, offset=offset)

                if not results['items']:
                    break

                # Get full track objects with popularity
                track_ids = [track['id'] for track in results['items'] if track['id']]

                if track_ids:
                    full_tracks = self.spotify.tracks(track_ids)['tracks']
                    tracks.extend(full_tracks)

                if len(results['items']) < limit:
                    break

                offset += limit

            return tracks

        except Exception as e:
            print(f"❌ Error fetching album tracks: {e}")
            return []

    def _calculate_popularity_rank(self, target_track: Dict, all_tracks: List[Dict]) -> int:
        """Calculate track's popularity rank within artist catalog."""
        # Sort tracks by popularity (descending)
        sorted_tracks = sorted(all_tracks, key=lambda x: x.get('popularity', 0), reverse=True)

        # Find rank of target track
        for i, track in enumerate(sorted_tracks):
            if track['id'] == target_track['id']:
                return i + 1  # 1-based ranking

        return len(all_tracks)  # If not found, rank as last

    def _calculate_momentum_score(self, target_track: Dict, all_tracks: List[Dict]) -> float:
        """
        Calculate momentum score based on popularity vs age ratio.

        Higher score = track is performing well for its age
        """
        try:
            target_popularity = target_track.get('popularity', 0)
            target_age_days = target_track.get('age_days')

            if target_age_days is None or target_age_days <= 0:
                return 0.5  # Neutral score for tracks without age data

            # Calculate average popularity per day for this artist's tracks
            valid_tracks = [t for t in all_tracks
                          if t.get('age_days') and t.get('age_days') > 0 and t.get('popularity', 0) > 0]

            if not valid_tracks:
                return 0.5

            # Calculate popularity per day for each track
            popularity_per_day_scores = []
            for track in valid_tracks:
                pop_per_day = track['popularity'] / max(track['age_days'], 1)
                popularity_per_day_scores.append(pop_per_day)

            if not popularity_per_day_scores:
                return 0.5

            # Calculate target track's popularity per day
            target_pop_per_day = target_popularity / max(target_age_days, 1)

            # Compare to artist's distribution
            avg_pop_per_day = np.mean(popularity_per_day_scores)
            std_pop_per_day = np.std(popularity_per_day_scores)

            if std_pop_per_day == 0:
                return 0.5

            # Z-score normalized to 0-1 range
            z_score = (target_pop_per_day - avg_pop_per_day) / std_pop_per_day
            momentum_score = 1 / (1 + np.exp(-z_score))  # Sigmoid normalization

            return min(max(momentum_score, 0.0), 1.0)

        except Exception as e:
            print(f"⚠️ Error calculating momentum score: {e}")
            return 0.5

    def _calculate_discovery_score(self, target_track: Dict, all_tracks: List[Dict]) -> float:
        """
        Calculate discovery score - how likely this is a hidden gem.

        Factors:
        - Low global popularity but decent in artist catalog
        - Recent release with climbing popularity
        - Not on the biggest albums
        """
        try:
            target_popularity = target_track.get('popularity', 0)
            target_age_days = target_track.get('age_days', 365)
            album_type = target_track.get('album_type', 'album')

            # Base score starts at 0.5
            discovery_score = 0.5

            # Factor 1: Low global popularity is good for discovery
            if target_popularity < 30:
                discovery_score += 0.2
            elif target_popularity < 50:
                discovery_score += 0.1

            # Factor 2: Recent releases get bonus
            if target_age_days < 30:  # Less than 30 days old
                discovery_score += 0.3
            elif target_age_days < 90:  # Less than 3 months old
                discovery_score += 0.2

            # Factor 3: Singles and EPs often contain hidden gems
            if album_type in ['single', 'ep']:
                discovery_score += 0.1

            # Factor 4: Track position in artist's popularity ranking
            popularity_rank = self._calculate_popularity_rank(target_track, all_tracks)
            total_tracks = len(all_tracks)

            # Sweet spot: not the #1 hit, but in top 50% of artist's catalog
            relative_rank = popularity_rank / total_tracks if total_tracks > 0 else 1

            if 0.1 < relative_rank < 0.5:  # In top 10-50% of artist's tracks
                discovery_score += 0.2
            elif 0.5 < relative_rank < 0.8:  # In 50-80% range
                discovery_score += 0.1

            return min(max(discovery_score, 0.0), 1.0)

        except Exception as e:
            print(f"⚠️ Error calculating discovery score: {e}")
            return 0.5

    def _calculate_relative_position(self, target_track: Dict, all_tracks: List[Dict]) -> Dict:
        """Calculate track's relative position in artist's catalog."""
        try:
            target_popularity = target_track.get('popularity', 0)

            # Get popularity distribution
            popularities = [t.get('popularity', 0) for t in all_tracks]
            popularities.sort()

            if not popularities:
                return {'percentile': 0.5, 'quartile': 2}

            # Calculate percentile
            below_count = sum(1 for p in popularities if p < target_popularity)
            percentile = below_count / len(popularities) if popularities else 0.5

            # Calculate quartile (1-4)
            if percentile >= 0.75:
                quartile = 4  # Top quartile
            elif percentile >= 0.5:
                quartile = 3
            elif percentile >= 0.25:
                quartile = 2
            else:
                quartile = 1  # Bottom quartile

            return {
                'percentile': percentile,
                'quartile': quartile
            }

        except Exception as e:
            print(f"⚠️ Error calculating relative position: {e}")
            return {'percentile': 0.5, 'quartile': 2}

    def _calculate_recommendation_weight(self, momentum_score: float,
                                       discovery_score: float,
                                       percentile: float) -> float:
        """
        Calculate overall recommendation weight for the track.

        This determines how strongly to recommend this track based on
        relative popularity factors.
        """
        try:
            # Combine scores with different weights
            weight = (
                0.4 * momentum_score +     # Trending tracks are important
                0.35 * discovery_score +   # Hidden gems are valuable
                0.25 * (1 - percentile)    # Lower percentile = more interesting
            )

            # Bonus for tracks that score well in multiple categories
            if momentum_score > 0.7 and discovery_score > 0.6:
                weight += 0.1  # Rising star bonus

            if discovery_score > 0.8 and percentile < 0.3:
                weight += 0.1  # Hidden gem bonus

            return min(max(weight, 0.0), 1.0)

        except Exception as e:
            print(f"⚠️ Error calculating recommendation weight: {e}")
            return 0.5

    def analyze_user_tracks_relative_popularity(self, user_tracks: List[str]) -> Dict:
        """
        Analyze relative popularity for a set of user tracks.

        Args:
            user_tracks: List of track IDs

        Returns:
            Dictionary with relative popularity analysis
        """
        try:
            print("🔍 Analyzing relative popularity for user tracks...")

            analyses = []
            artist_track_map = {}

            # Group tracks by artist for efficient processing
            for track_id in user_tracks:
                try:
                    track = self.spotify.track(track_id)
                    artist_id = track['artists'][0]['id']

                    if artist_id not in artist_track_map:
                        artist_track_map[artist_id] = []
                    artist_track_map[artist_id].append(track_id)

                except Exception as e:
                    print(f"⚠️ Error processing track {track_id}: {e}")
                    continue

            # Analyze each artist's tracks
            for artist_id, track_ids in artist_track_map.items():
                for track_id in track_ids:
                    analysis = self.calculate_relative_popularity_score(track_id, artist_id)
                    if 'error' not in analysis:
                        analyses.append(analysis)

            # Generate summary statistics
            summary = self._generate_relative_popularity_summary(analyses)

            return {
                'track_analyses': analyses,
                'summary': summary,
                'total_tracks_analyzed': len(analyses),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error analyzing user tracks relative popularity: {e}")
            return {'error': str(e)}

    def _generate_relative_popularity_summary(self, analyses: List[Dict]) -> Dict:
        """Generate summary statistics from relative popularity analyses."""
        if not analyses:
            return {}

        try:
            momentum_scores = [a['momentum_score'] for a in analyses]
            discovery_scores = [a['discovery_score'] for a in analyses]
            recommendation_weights = [a['recommendation_weight'] for a in analyses]

            # Find interesting tracks
            rising_stars = [a for a in analyses if a.get('is_rising_star', False)]
            hidden_gems = [a for a in analyses if a.get('is_hidden_gem', False)]

            return {
                'avg_momentum_score': np.mean(momentum_scores),
                'avg_discovery_score': np.mean(discovery_scores),
                'avg_recommendation_weight': np.mean(recommendation_weights),
                'rising_stars_count': len(rising_stars),
                'hidden_gems_count': len(hidden_gems),
                'top_momentum_tracks': sorted(analyses, key=lambda x: x['momentum_score'], reverse=True)[:5],
                'top_discovery_tracks': sorted(analyses, key=lambda x: x['discovery_score'], reverse=True)[:5],
                'user_discovery_tendency': 'high' if np.mean(discovery_scores) > 0.7 else 'medium' if np.mean(discovery_scores) > 0.5 else 'low'
            }

        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
            return {}


def enhance_recommendations_with_relative_popularity(recommendations: List[Dict],
                                                   spotify_client,
                                                   analyzer: RelativePopularityAnalyzer = None) -> List[Dict]:
    """
    Enhance existing recommendations with relative popularity scores.

    Args:
        recommendations: List of recommendation dictionaries
        spotify_client: Spotify client
        analyzer: Optional pre-initialized analyzer

    Returns:
        Enhanced recommendations with relative popularity data
    """
    if analyzer is None:
        analyzer = RelativePopularityAnalyzer(spotify_client, None)

    enhanced_recs = []

    for rec in recommendations:
        try:
            track_id = rec.get('id')
            if not track_id:
                enhanced_recs.append(rec)
                continue

            # Get artist ID
            track = spotify_client.track(track_id)
            artist_id = track['artists'][0]['id']

            # Calculate relative popularity
            rel_pop = analyzer.calculate_relative_popularity_score(track_id, artist_id)

            if 'error' not in rel_pop:
                # Add relative popularity data to recommendation
                enhanced_rec = rec.copy()
                enhanced_rec['relative_popularity'] = rel_pop
                enhanced_rec['discovery_potential'] = rel_pop.get('discovery_score', 0.5)
                enhanced_rec['momentum'] = rel_pop.get('momentum_score', 0.5)

                # Adjust recommendation score based on relative popularity
                if 'similarity_score' in enhanced_rec:
                    rel_weight = rel_pop.get('recommendation_weight', 0.5)
                    enhanced_rec['enhanced_score'] = (
                        0.7 * enhanced_rec['similarity_score'] +
                        0.3 * rel_weight
                    )

                enhanced_recs.append(enhanced_rec)
            else:
                enhanced_recs.append(rec)

        except Exception as e:
            print(f"⚠️ Error enhancing recommendation: {e}")
            enhanced_recs.append(rec)

    # Re-sort by enhanced score if available
    enhanced_recs.sort(key=lambda x: x.get('enhanced_score', x.get('similarity_score', 0)), reverse=True)

    return enhanced_recs
