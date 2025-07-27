"""
Advanced Music Recommendation Engine

Uses extracted audio features to generate sophisticated music recommendations.
Implements multiple recommendation strategies:
- Content-based filtering using audio features
- Collaborative filtering approaches
- Hybrid methods combining multiple signals
- Advanced similarity algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json
from datetime import datetime, timedelta


class MusicRecommender:
    """Advanced music recommendation system using multiple algorithms."""

    def __init__(self, features_df: pd.DataFrame, tracks_df: pd.DataFrame):
        """
        Initialize the recommender with feature and track data.

        Args:
            features_df: DataFrame with extracted audio features
            tracks_df: DataFrame with track metadata
        """
        self.features_df = features_df.copy()
        self.tracks_df = tracks_df.copy()
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.feature_matrix = None
        self.track_clusters = None
        self.knn_model = None

        self._prepare_data()

    def _prepare_data(self):
        """Prepare and normalize feature data for recommendation algorithms."""
        print("🔧 Preparing data for recommendations...")

        # Merge features with track metadata
        self.data = pd.merge(
            self.tracks_df,
            self.features_df,
            left_on='id',
            right_on='track_id',
            how='inner'
        )

        # Select numerical features for similarity calculation
        feature_columns = [col for col in self.features_df.columns
                         if col not in ['track_id', 'analysis_method', 'key', 'scale']]

        # Handle missing values
        self.feature_matrix = self.features_df[feature_columns].fillna(0)

        # Normalize features
        self.normalized_features = self.scaler.fit_transform(self.feature_matrix)

        # Create similarity matrix
        self.similarity_matrix = cosine_similarity(self.normalized_features)

        print(f"✅ Prepared {len(self.data)} tracks with {len(feature_columns)} features")

    def find_similar_tracks(self, track_id: str, n_recommendations: int = 10,
                          algorithm: str = 'cosine') -> List[Dict]:
        """
        Find tracks similar to a given track using various similarity algorithms.

        Args:
            track_id: ID of the seed track
            n_recommendations: Number of recommendations to return
            algorithm: Similarity algorithm ('cosine', 'euclidean', 'hybrid')

        Returns:
            List of recommended track dictionaries with similarity scores
        """
        try:
            # Find track index
            track_indices = self.features_df[self.features_df['track_id'] == track_id].index
            if len(track_indices) == 0:
                print(f"❌ Track {track_id} not found in features")
                return []

            track_idx = track_indices[0]

            if algorithm == 'cosine':
                similarities = self.similarity_matrix[track_idx]
            elif algorithm == 'euclidean':
                distances = euclidean_distances([self.normalized_features[track_idx]],
                                              self.normalized_features)[0]
                similarities = 1 / (1 + distances)  # Convert distance to similarity
            elif algorithm == 'hybrid':
                # Combine cosine and euclidean similarities
                cosine_sim = self.similarity_matrix[track_idx]
                distances = euclidean_distances([self.normalized_features[track_idx]],
                                              self.normalized_features)[0]
                euclidean_sim = 1 / (1 + distances)
                similarities = 0.7 * cosine_sim + 0.3 * euclidean_sim
            else:
                similarities = self.similarity_matrix[track_idx]

            # Get top similar tracks (excluding the seed track itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

            recommendations = []
            for idx in similar_indices:
                if idx < len(self.data):
                    track_info = self.data.iloc[idx].to_dict()
                    track_info['similarity_score'] = float(similarities[idx])
                    track_info['recommendation_method'] = f'content_{algorithm}'
                    recommendations.append(track_info)

            return recommendations

        except Exception as e:
            print(f"❌ Error finding similar tracks: {e}")
            return []

    def get_cluster_recommendations(self, track_id: str, n_recommendations: int = 10,
                                  n_clusters: int = 20) -> List[Dict]:
        """
        Get recommendations based on track clustering.

        Args:
            track_id: ID of the seed track
            n_recommendations: Number of recommendations to return
            n_clusters: Number of clusters for KMeans

        Returns:
            List of recommended tracks from the same cluster
        """
        try:
            # Perform clustering if not already done
            if self.track_clusters is None:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                self.track_clusters = kmeans.fit_predict(self.normalized_features)

            # Find the cluster of the seed track
            track_indices = self.features_df[self.features_df['track_id'] == track_id].index
            if len(track_indices) == 0:
                return []

            track_idx = track_indices[0]
            seed_cluster = self.track_clusters[track_idx]

            # Find all tracks in the same cluster
            cluster_indices = np.where(self.track_clusters == seed_cluster)[0]

            # Remove the seed track itself
            cluster_indices = cluster_indices[cluster_indices != track_idx]

            # Get similarity scores within the cluster
            similarities = self.similarity_matrix[track_idx][cluster_indices]

            # Sort by similarity and take top N
            sorted_indices = cluster_indices[np.argsort(similarities)[::-1]]
            top_indices = sorted_indices[:n_recommendations]

            recommendations = []
            for idx in top_indices:
                if idx < len(self.data):
                    track_info = self.data.iloc[idx].to_dict()
                    track_info['similarity_score'] = float(self.similarity_matrix[track_idx][idx])
                    track_info['cluster_id'] = int(seed_cluster)
                    track_info['recommendation_method'] = 'cluster_based'
                    recommendations.append(track_info)

            return recommendations

        except Exception as e:
            print(f"❌ Error in cluster recommendations: {e}")
            return []

    def get_feature_based_recommendations(self, track_id: str, feature_weights: Dict[str, float] = None,
                                        n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations based on specific audio features with custom weights.

        Args:
            track_id: ID of the seed track
            feature_weights: Dictionary of feature names and their weights
            n_recommendations: Number of recommendations to return

        Returns:
            List of recommended tracks
        """
        try:
            # Default feature weights if not provided
            if feature_weights is None:
                feature_weights = {
                    'tempo_essentia': 0.2,
                    'valence': 0.15,
                    'energy': 0.15,
                    'danceability': 0.1,
                    'acousticness': 0.1,
                    'spectral_centroid_mean': 0.15,
                    'mfcc_0_mean': 0.15
                }

            # Find track features
            track_features = self.features_df[self.features_df['track_id'] == track_id]
            if len(track_features) == 0:
                return []

            seed_features = track_features.iloc[0]

            # Calculate weighted similarity
            similarities = []
            for _, track in self.features_df.iterrows():
                if track['track_id'] == track_id:
                    similarities.append(0)  # Don't recommend the same track
                    continue

                weighted_distance = 0
                total_weight = 0

                for feature, weight in feature_weights.items():
                    if feature in seed_features and feature in track:
                        if pd.notna(seed_features[feature]) and pd.notna(track[feature]):
                            # Normalize the difference
                            feature_diff = abs(seed_features[feature] - track[feature])
                            feature_range = self.features_df[feature].max() - self.features_df[feature].min()
                            if feature_range > 0:
                                normalized_diff = feature_diff / feature_range
                                weighted_distance += weight * normalized_diff
                                total_weight += weight

                if total_weight > 0:
                    similarity = 1 - (weighted_distance / total_weight)
                else:
                    similarity = 0

                similarities.append(similarity)

            # Get top recommendations
            similarities = np.array(similarities)
            top_indices = np.argsort(similarities)[::-1][:n_recommendations]

            recommendations = []
            for idx in top_indices:
                if idx < len(self.data):
                    track_info = self.data.iloc[idx].to_dict()
                    track_info['similarity_score'] = float(similarities[idx])
                    track_info['recommendation_method'] = 'weighted_features'
                    track_info['feature_weights'] = feature_weights
                    recommendations.append(track_info)

            return recommendations

        except Exception as e:
            print(f"❌ Error in feature-based recommendations: {e}")
            return []

    def get_mood_based_recommendations(self, mood: str, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations based on mood/energy level.

        Args:
            mood: Mood descriptor ('energetic', 'chill', 'happy', 'sad', 'party', 'focus')
            n_recommendations: Number of recommendations to return

        Returns:
            List of recommended tracks matching the mood
        """
        try:
            # Define mood criteria using audio features
            mood_criteria = {
                'energetic': {
                    'energy_min': 0.7, 'tempo_min': 120, 'danceability_min': 0.6
                },
                'chill': {
                    'energy_max': 0.5, 'tempo_max': 100, 'acousticness_min': 0.3
                },
                'happy': {
                    'valence_min': 0.6, 'energy_min': 0.5, 'danceability_min': 0.5
                },
                'sad': {
                    'valence_max': 0.4, 'energy_max': 0.4, 'acousticness_min': 0.3
                },
                'party': {
                    'energy_min': 0.8, 'danceability_min': 0.7, 'tempo_min': 110
                },
                'focus': {
                    'instrumentalness_min': 0.5, 'energy_max': 0.6, 'speechiness_max': 0.1
                }
            }

            if mood not in mood_criteria:
                print(f"❌ Unknown mood: {mood}")
                return []

            criteria = mood_criteria[mood]

            # Filter tracks based on mood criteria
            filtered_data = self.data.copy()

            for criterion, value in criteria.items():
                feature = criterion.replace('_min', '').replace('_max', '')

                if criterion.endswith('_min'):
                    filtered_data = filtered_data[filtered_data[feature] >= value]
                elif criterion.endswith('_max'):
                    filtered_data = filtered_data[filtered_data[feature] <= value]

            # Sort by relevance (combination of criteria)
            if len(filtered_data) > 0:
                # Calculate mood score
                mood_scores = []
                for _, track in filtered_data.iterrows():
                    score = 0
                    for criterion, value in criteria.items():
                        feature = criterion.replace('_min', '').replace('_max', '')
                        if feature in track and pd.notna(track[feature]):
                            if criterion.endswith('_min'):
                                score += max(0, track[feature] - value)
                            elif criterion.endswith('_max'):
                                score += max(0, value - track[feature])
                    mood_scores.append(score)

                filtered_data['mood_score'] = mood_scores
                filtered_data = filtered_data.sort_values('mood_score', ascending=False)

                # Return top recommendations
                recommendations = []
                for _, track in filtered_data.head(n_recommendations).iterrows():
                    track_dict = track.to_dict()
                    track_dict['recommendation_method'] = f'mood_{mood}'
                    track_dict['mood_criteria'] = criteria
                    recommendations.append(track_dict)

                return recommendations
            else:
                print(f"❌ No tracks found matching {mood} criteria")
                return []

        except Exception as e:
            print(f"❌ Error in mood-based recommendations: {e}")
            return []

    def create_playlist_recommendations(self, seed_tracks: List[str],
                                      playlist_length: int = 20,
                                      diversity_factor: float = 0.3) -> List[Dict]:
        """
        Create a balanced playlist from multiple seed tracks.

        Args:
            seed_tracks: List of track IDs to base recommendations on
            playlist_length: Target number of tracks in playlist
            diversity_factor: How much to prioritize diversity (0-1)

        Returns:
            List of recommended tracks for playlist
        """
        try:
            all_recommendations = []

            # Get recommendations from each seed track
            for seed_track in seed_tracks:
                similar_tracks = self.find_similar_tracks(
                    seed_track,
                    n_recommendations=playlist_length // len(seed_tracks) + 5
                )
                all_recommendations.extend(similar_tracks)

            # Remove duplicates while keeping track of scores
            track_scores = {}
            for rec in all_recommendations:
                track_id = rec['id']
                if track_id not in track_scores:
                    track_scores[track_id] = {
                        'track': rec,
                        'scores': [],
                        'avg_score': 0
                    }
                track_scores[track_id]['scores'].append(rec['similarity_score'])

            # Calculate average scores and diversity penalty
            final_recommendations = []
            selected_features = []

            for track_id, data in track_scores.items():
                data['avg_score'] = np.mean(data['scores'])

                # Add diversity penalty if enabled
                if diversity_factor > 0 and len(selected_features) > 0:
                    track_features = self.features_df[
                        self.features_df['track_id'] == track_id
                    ]
                    if len(track_features) > 0:
                        track_idx = track_features.index[0]

                        # Calculate similarity to already selected tracks
                        diversities = []
                        for selected_idx in selected_features:
                            similarity = self.similarity_matrix[track_idx][selected_idx]
                            diversities.append(similarity)

                        avg_similarity = np.mean(diversities)
                        diversity_penalty = diversity_factor * avg_similarity
                        data['final_score'] = data['avg_score'] - diversity_penalty
                    else:
                        data['final_score'] = data['avg_score']
                else:
                    data['final_score'] = data['avg_score']

                final_recommendations.append(data)

            # Sort by final score and select top tracks
            final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)

            playlist = []
            for rec in final_recommendations[:playlist_length]:
                track_info = rec['track'].copy()
                track_info['playlist_score'] = rec['final_score']
                track_info['recommendation_method'] = 'playlist_balanced'
                playlist.append(track_info)

                # Update selected features for diversity calculation
                track_features = self.features_df[
                    self.features_df['track_id'] == rec['track']['id']
                ]
                if len(track_features) > 0:
                    selected_features.append(track_features.index[0])

            return playlist

        except Exception as e:
            print(f"❌ Error creating playlist recommendations: {e}")
            return []

    def analyze_user_taste_profile(self, user_tracks: List[str]) -> Dict:
        """
        Analyze a user's taste profile based on their listening history.

        Args:
            user_tracks: List of track IDs from user's listening history

        Returns:
            Dictionary containing taste profile analysis
        """
        try:
            # Get features for user tracks
            user_features = self.features_df[
                self.features_df['track_id'].isin(user_tracks)
            ]

            if len(user_features) == 0:
                return {}

            # Calculate statistical measures for each feature
            taste_profile = {}

            numeric_columns = user_features.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'track_id']

            for column in numeric_columns:
                values = user_features[column].dropna()
                if len(values) > 0:
                    taste_profile[column] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }

            # Add categorical features
            if 'key' in user_features.columns:
                key_counts = user_features['key'].value_counts()
                taste_profile['preferred_keys'] = key_counts.to_dict()

            if 'scale' in user_features.columns:
                scale_counts = user_features['scale'].value_counts()
                taste_profile['preferred_scales'] = scale_counts.to_dict()

            # Overall taste summary
            taste_profile['summary'] = {
                'total_tracks_analyzed': len(user_features),
                'avg_energy': taste_profile.get('energy', {}).get('mean', 0),
                'avg_valence': taste_profile.get('valence', {}).get('mean', 0),
                'avg_tempo': taste_profile.get('tempo_essentia', {}).get('mean', 0),
                'analysis_date': datetime.now().isoformat()
            }

            return taste_profile

        except Exception as e:
            print(f"❌ Error analyzing taste profile: {e}")
            return {}

    def get_personalized_recommendations(self, user_tracks: List[str],
                                       n_recommendations: int = 20) -> List[Dict]:
        """
        Get personalized recommendations based on user's listening history.

        Args:
            user_tracks: List of track IDs from user's listening history
            n_recommendations: Number of recommendations to return

        Returns:
            List of personalized recommendations
        """
        try:
            # Analyze user taste profile
            taste_profile = self.analyze_user_taste_profile(user_tracks)

            if not taste_profile:
                return []

            # Get recommendations from multiple methods
            all_recs = []

            # Method 1: Similar to user's top tracks
            for track_id in user_tracks[:5]:  # Use top 5 as seeds
                similar = self.find_similar_tracks(track_id, n_recommendations=5)
                all_recs.extend(similar)

            # Method 2: Cluster-based recommendations
            for track_id in user_tracks[:3]:
                cluster_recs = self.get_cluster_recommendations(track_id, n_recommendations=5)
                all_recs.extend(cluster_recs)

            # Method 3: Feature-based with user preferences
            feature_weights = self._extract_feature_weights_from_profile(taste_profile)
            if user_tracks:
                feature_recs = self.get_feature_based_recommendations(
                    user_tracks[0], feature_weights, n_recommendations=10
                )
                all_recs.extend(feature_recs)

            # Remove duplicates and user's existing tracks
            seen_tracks = set(user_tracks)
            unique_recs = []
            for rec in all_recs:
                if rec['id'] not in seen_tracks:
                    unique_recs.append(rec)
                    seen_tracks.add(rec['id'])

            # Score and rank recommendations
            scored_recs = self._score_recommendations_for_user(unique_recs, taste_profile)

            return scored_recs[:n_recommendations]

        except Exception as e:
            print(f"❌ Error getting personalized recommendations: {e}")
            return []

    def _extract_feature_weights_from_profile(self, taste_profile: Dict) -> Dict[str, float]:
        """Extract feature weights based on user's taste profile variance."""
        weights = {}

        # Features with lower variance get higher weights (more consistent preferences)
        for feature, stats in taste_profile.items():
            if isinstance(stats, dict) and 'std' in stats and 'mean' in stats:
                if stats['mean'] != 0:  # Avoid division by zero
                    # Coefficient of variation (lower = more consistent)
                    cv = stats['std'] / abs(stats['mean'])
                    weight = 1 / (1 + cv)  # Higher weight for lower variance
                    weights[feature] = weight

        # Normalize weights
        if weights:
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

        return weights

    def _score_recommendations_for_user(self, recommendations: List[Dict],
                                      taste_profile: Dict) -> List[Dict]:
        """Score recommendations based on how well they match user's taste profile."""
        try:
            for rec in recommendations:
                track_id = rec['id']
                track_features = self.features_df[
                    self.features_df['track_id'] == track_id
                ]

                if len(track_features) == 0:
                    rec['personalization_score'] = 0.5
                    continue

                track_feature = track_features.iloc[0]

                # Calculate how well this track matches user's preferences
                match_scores = []

                for feature, stats in taste_profile.items():
                    if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
                        if feature in track_feature and pd.notna(track_feature[feature]):
                            user_mean = stats['mean']
                            user_std = stats['std'] if stats['std'] > 0 else 1
                            track_value = track_feature[feature]

                            # Calculate z-score and convert to similarity
                            z_score = abs(track_value - user_mean) / user_std
                            similarity = np.exp(-z_score)  # Gaussian similarity
                            match_scores.append(similarity)

                if match_scores:
                    rec['personalization_score'] = np.mean(match_scores)
                else:
                    rec['personalization_score'] = 0.5

                # Combine with existing similarity score
                if 'similarity_score' in rec:
                    rec['final_score'] = 0.6 * rec['personalization_score'] + 0.4 * rec['similarity_score']
                else:
                    rec['final_score'] = rec['personalization_score']

            # Sort by final score
            recommendations.sort(key=lambda x: x.get('final_score', 0), reverse=True)

            return recommendations

        except Exception as e:
            print(f"❌ Error scoring recommendations: {e}")
            return recommendations
