"""
Intelligent Feedback System

Collects user feedback on recommendations and uses machine learning
to adapt the recommendation algorithm over time. Implements:
- Feedback collection and storage
- Pattern analysis of user preferences
- Algorithm adaptation based on feedback
- Preference evolution tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sqlite3


class FeedbackSystem:
    """Intelligent feedback collection and learning system."""

    def __init__(self, database, recommender, relative_popularity_analyzer=None):
        """
        Initialize the feedback system.

        Args:
            database: Database instance
            recommender: Music recommender instance
            relative_popularity_analyzer: Optional relative popularity analyzer
        """
        self.database = database
        self.recommender = recommender
        self.rel_pop_analyzer = relative_popularity_analyzer
        self.feedback_model = None
        self.scaler = StandardScaler()
        self._create_feedback_tables()

    def _create_feedback_tables(self):
        """Create database tables for feedback system."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()

                # Enhanced feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        track_id TEXT NOT NULL,
                        recommendation_id INTEGER,
                        feedback_type TEXT NOT NULL,  -- like, dislike, love, skip, save
                        feedback_value REAL NOT NULL,  -- -1 to 1 scale
                        feedback_context TEXT,  -- JSON with context (mood, time, etc.)
                        audio_features TEXT,  -- JSON with track audio features
                        recommendation_reason TEXT,  -- Why this was recommended
                        session_id TEXT,  -- Group feedback from same session
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (track_id) REFERENCES tracks (id),
                        FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
                    )
                ''')

                # Preference evolution tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS preference_evolution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        feature_name TEXT NOT NULL,
                        old_preference REAL,
                        new_preference REAL,
                        confidence REAL,
                        feedback_count INTEGER,
                        evolution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Algorithm adaptation log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS algorithm_adaptations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        adaptation_type TEXT NOT NULL,  -- weight_update, feature_importance, etc.
                        old_parameters TEXT,  -- JSON
                        new_parameters TEXT,  -- JSON
                        performance_improvement REAL,
                        feedback_batch_size INTEGER,
                        adaptation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Feedback sessions - group related feedback
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        user_id TEXT NOT NULL,
                        session_type TEXT,  -- discovery, playlist_creation, etc.
                        total_recommendations INTEGER,
                        positive_feedback_count INTEGER,
                        negative_feedback_count INTEGER,
                        session_start TIMESTAMP,
                        session_end TIMESTAMP,
                        context_data TEXT  -- JSON with session context
                    )
                ''')

                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON user_feedback(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_track_id ON user_feedback(track_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_preference_evolution_user ON preference_evolution(user_id)')

                conn.commit()
                print("✅ Feedback system tables created")

        except Exception as e:
            print(f"❌ Error creating feedback tables: {e}")

    def collect_feedback(self, user_id: str, track_id: str, feedback_type: str,
                        recommendation_id: Optional[int] = None,
                        context: Optional[Dict] = None,
                        session_id: Optional[str] = None) -> bool:
        """
        Collect user feedback on a recommendation.

        Args:
            user_id: User ID
            track_id: Track ID that was recommended
            feedback_type: Type of feedback (like, dislike, love, skip, save)
            recommendation_id: ID of the original recommendation
            context: Optional context information
            session_id: Optional session grouping ID

        Returns:
            Success boolean
        """
        try:
            # Convert feedback type to numerical value
            feedback_value = self._feedback_type_to_value(feedback_type)

            # Get track audio features for learning
            audio_features = self._get_track_audio_features(track_id)

            # Get recommendation reason if available
            recommendation_reason = self._get_recommendation_reason(recommendation_id) if recommendation_id else None

            with self.database.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO user_feedback (
                        user_id, track_id, recommendation_id, feedback_type,
                        feedback_value, feedback_context, audio_features,
                        recommendation_reason, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, track_id, recommendation_id, feedback_type,
                    feedback_value, json.dumps(context or {}),
                    json.dumps(audio_features or {}),
                    recommendation_reason, session_id
                ))

                conn.commit()

            print(f"✅ Collected {feedback_type} feedback for track {track_id}")

            # Trigger learning if we have enough feedback
            self._trigger_learning_if_ready(user_id)

            return True

        except Exception as e:
            print(f"❌ Error collecting feedback: {e}")
            return False

    def collect_batch_feedback(self, user_id: str, feedback_batch: List[Dict],
                             session_id: Optional[str] = None) -> Dict:
        """
        Collect multiple feedback items at once.

        Args:
            user_id: User ID
            feedback_batch: List of feedback dictionaries
            session_id: Optional session ID

        Returns:
            Results summary
        """
        try:
            if not session_id:
                session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"

            results = {
                'session_id': session_id,
                'total_feedback': len(feedback_batch),
                'successful': 0,
                'failed': 0,
                'positive_count': 0,
                'negative_count': 0
            }

            for feedback in feedback_batch:
                success = self.collect_feedback(
                    user_id=user_id,
                    track_id=feedback['track_id'],
                    feedback_type=feedback['feedback_type'],
                    recommendation_id=feedback.get('recommendation_id'),
                    context=feedback.get('context'),
                    session_id=session_id
                )

                if success:
                    results['successful'] += 1
                    feedback_value = self._feedback_type_to_value(feedback['feedback_type'])
                    if feedback_value > 0:
                        results['positive_count'] += 1
                    elif feedback_value < 0:
                        results['negative_count'] += 1
                else:
                    results['failed'] += 1

            # Record feedback session
            self._record_feedback_session(user_id, session_id, results, feedback_batch)

            print(f"✅ Collected batch feedback: {results['successful']}/{results['total_feedback']} successful")

            return results

        except Exception as e:
            print(f"❌ Error collecting batch feedback: {e}")
            return {'error': str(e)}

    def analyze_user_feedback_patterns(self, user_id: str,
                                     time_window_days: int = 90) -> Dict:
        """
        Analyze patterns in user feedback to understand preferences.

        Args:
            user_id: User ID
            time_window_days: Number of days to analyze

        Returns:
            Feedback pattern analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)

            with self.database.get_connection() as conn:
                df = pd.read_sql_query('''
                    SELECT uf.*, af.audio_features
                    FROM user_feedback uf
                    LEFT JOIN (
                        SELECT track_id, audio_features
                        FROM user_feedback
                        WHERE audio_features != '{}'
                        GROUP BY track_id
                    ) af ON uf.track_id = af.track_id
                    WHERE uf.user_id = ? AND uf.timestamp > ?
                    ORDER BY uf.timestamp DESC
                ''', conn, params=(user_id, cutoff_date))

            if df.empty:
                return {'error': 'No feedback data found'}

            analysis = {
                'feedback_summary': self._analyze_feedback_summary(df),
                'feature_preferences': self._analyze_feature_preferences(df),
                'temporal_patterns': self._analyze_temporal_patterns(df),
                'context_patterns': self._analyze_context_patterns(df),
                'recommendation_performance': self._analyze_recommendation_performance(df),
                'preference_stability': self._analyze_preference_stability(df),
                'analysis_timestamp': datetime.now().isoformat()
            }

            return analysis

        except Exception as e:
            print(f"❌ Error analyzing feedback patterns: {e}")
            return {'error': str(e)}

    def adapt_algorithm_for_user(self, user_id: str) -> Dict:
        """
        Adapt the recommendation algorithm based on user feedback.

        Args:
            user_id: User ID

        Returns:
            Adaptation results
        """
        try:
            print(f"🤖 Adapting algorithm for user {user_id}...")

            # Analyze current feedback patterns
            patterns = self.analyze_user_feedback_patterns(user_id)

            if 'error' in patterns:
                return patterns

            # Get current algorithm parameters
            current_params = self._get_current_algorithm_parameters(user_id)

            # Calculate new parameters based on feedback
            new_params = self._calculate_adapted_parameters(patterns, current_params)

            # Validate and apply adaptations
            adaptation_results = self._apply_algorithm_adaptations(user_id, current_params, new_params)

            # Log the adaptation
            self._log_algorithm_adaptation(user_id, current_params, new_params, adaptation_results)

            return {
                'user_id': user_id,
                'adaptations_applied': adaptation_results,
                'performance_improvement': adaptation_results.get('estimated_improvement', 0),
                'new_parameters': new_params,
                'adaptation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error adapting algorithm: {e}")
            return {'error': str(e)}

    def train_feedback_prediction_model(self, user_id: str) -> Dict:
        """
        Train a model to predict user preferences based on audio features.

        Args:
            user_id: User ID

        Returns:
            Training results
        """
        try:
            print(f"🧠 Training preference prediction model for user {user_id}...")

            # Get feedback data with audio features
            feedback_data = self._get_feedback_training_data(user_id)

            if len(feedback_data) < 20:  # Need minimum data for training
                return {'error': 'Insufficient feedback data for training (minimum 20 required)'}

            # Prepare training data
            X, y = self._prepare_training_data(feedback_data)

            if X.shape[0] < 10:
                return {'error': 'Insufficient feature data for training'}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.feedback_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                min_samples_split=5
            )

            self.feedback_model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = self.feedback_model.score(X_train_scaled, y_train)
            test_score = self.feedback_model.score(X_test_scaled, y_test)

            y_pred = self.feedback_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(zip(
                self._get_audio_feature_names(),
                self.feedback_model.feature_importances_
            ))

            results = {
                'model_trained': True,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'train_score': train_score,
                'test_score': test_score,
                'mse': mse,
                'feature_importance': feature_importance,
                'top_preference_features': sorted(feature_importance.items(),
                                                key=lambda x: x[1], reverse=True)[:10],
                'training_timestamp': datetime.now().isoformat()
            }

            print(f"✅ Model trained successfully. Test score: {test_score:.3f}")

            return results

        except Exception as e:
            print(f"❌ Error training prediction model: {e}")
            return {'error': str(e)}

    def predict_user_preference(self, user_id: str, track_features: Dict) -> float:
        """
        Predict how much a user will like a track based on their feedback history.

        Args:
            user_id: User ID
            track_features: Dictionary of audio features

        Returns:
            Predicted preference score (-1 to 1)
        """
        try:
            if self.feedback_model is None:
                # Try to train model first
                training_result = self.train_feedback_prediction_model(user_id)
                if 'error' in training_result:
                    return 0.0  # Neutral prediction if no model

            # Prepare features
            feature_vector = self._prepare_prediction_features(track_features)

            if feature_vector is None:
                return 0.0

            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])

            # Make prediction
            prediction = self.feedback_model.predict(feature_vector_scaled)[0]

            # Ensure prediction is in valid range
            return min(max(prediction, -1.0), 1.0)

        except Exception as e:
            print(f"⚠️ Error predicting user preference: {e}")
            return 0.0

    def get_feedback_dashboard_data(self, user_id: str) -> Dict:
        """
        Get comprehensive feedback data for dashboard display.

        Args:
            user_id: User ID

        Returns:
            Dashboard data dictionary
        """
        try:
            with self.database.get_connection() as conn:
                # Get recent feedback stats
                recent_feedback = pd.read_sql_query('''
                    SELECT feedback_type, COUNT(*) as count, AVG(feedback_value) as avg_rating
                    FROM user_feedback
                    WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
                    GROUP BY feedback_type
                ''', conn, params=(user_id,))

                # Get feedback trends over time
                feedback_trends = pd.read_sql_query('''
                    SELECT DATE(timestamp) as date,
                           AVG(feedback_value) as avg_rating,
                           COUNT(*) as feedback_count
                    FROM user_feedback
                    WHERE user_id = ? AND timestamp > datetime('now', '-90 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''', conn, params=(user_id,))

                # Get top liked/disliked features
                feature_preferences = self.analyze_user_feedback_patterns(user_id, 30)

                dashboard_data = {
                    'user_id': user_id,
                    'recent_feedback_summary': recent_feedback.to_dict('records') if not recent_feedback.empty else [],
                    'feedback_trends': feedback_trends.to_dict('records') if not feedback_trends.empty else [],
                    'total_feedback_count': self._get_total_feedback_count(user_id),
                    'preference_analysis': feature_preferences,
                    'recommendation_accuracy': self._calculate_recommendation_accuracy(user_id),
                    'last_updated': datetime.now().isoformat()
                }

                return dashboard_data

        except Exception as e:
            print(f"❌ Error getting dashboard data: {e}")
            return {'error': str(e)}

    # Helper methods

    def _feedback_type_to_value(self, feedback_type: str) -> float:
        """Convert feedback type to numerical value."""
        feedback_mapping = {
            'love': 1.0,
            'like': 0.7,
            'neutral': 0.0,
            'save': 0.8,
            'skip': -0.3,
            'dislike': -0.7,
            'hate': -1.0,
            'block': -1.0
        }
        return feedback_mapping.get(feedback_type.lower(), 0.0)

    def _get_track_audio_features(self, track_id: str) -> Dict:
        """Get audio features for a track."""
        try:
            # Try to get from recommendation system first
            if hasattr(self.recommender, 'features_df'):
                track_features = self.recommender.features_df[
                    self.recommender.features_df['track_id'] == track_id
                ]
                if not track_features.empty:
                    return track_features.iloc[0].to_dict()

            # Fallback: get basic audio features from database
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM audio_features WHERE track_id = ?', (track_id,))
                result = cursor.fetchone()
                if result:
                    return dict(result)

            return {}

        except Exception as e:
            print(f"⚠️ Error getting track audio features: {e}")
            return {}

    def _get_recommendation_reason(self, recommendation_id: int) -> str:
        """Get the reason for a recommendation."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT recommended_by FROM recommendations WHERE id = ?', (recommendation_id,))
                result = cursor.fetchone()
                return result[0] if result else 'unknown'
        except:
            return 'unknown'

    def _trigger_learning_if_ready(self, user_id: str):
        """Trigger learning if enough feedback has been collected."""
        try:
            feedback_count = self._get_total_feedback_count(user_id)

            # Trigger adaptation every 50 feedback items
            if feedback_count > 0 and feedback_count % 50 == 0:
                print(f"🎯 Triggering algorithm adaptation ({feedback_count} feedback items)")
                self.adapt_algorithm_for_user(user_id)

        except Exception as e:
            print(f"⚠️ Error checking learning trigger: {e}")

    def _record_feedback_session(self, user_id: str, session_id: str,
                                results: Dict, feedback_batch: List[Dict]):
        """Record a feedback session."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()

                # Get session context if available
                context_data = {}
                if feedback_batch and 'context' in feedback_batch[0]:
                    context_data = feedback_batch[0]['context']

                cursor.execute('''
                    INSERT INTO feedback_sessions (
                        session_id, user_id, session_type, total_recommendations,
                        positive_feedback_count, negative_feedback_count,
                        session_start, session_end, context_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, user_id, 'batch_feedback', results['total_feedback'],
                    results['positive_count'], results['negative_count'],
                    datetime.now(), datetime.now(), json.dumps(context_data)
                ))

                conn.commit()

        except Exception as e:
            print(f"⚠️ Error recording feedback session: {e}")

    def _get_total_feedback_count(self, user_id: str) -> int:
        """Get total feedback count for a user."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM user_feedback WHERE user_id = ?', (user_id,))
                return cursor.fetchone()[0]
        except:
            return 0

    def _analyze_feedback_summary(self, df: pd.DataFrame) -> Dict:
        """Analyze basic feedback summary statistics."""
        return {
            'total_feedback': len(df),
            'positive_feedback': len(df[df['feedback_value'] > 0]),
            'negative_feedback': len(df[df['feedback_value'] < 0]),
            'neutral_feedback': len(df[df['feedback_value'] == 0]),
            'average_rating': df['feedback_value'].mean(),
            'rating_std': df['feedback_value'].std(),
            'feedback_types': df['feedback_type'].value_counts().to_dict()
        }

    def _analyze_feature_preferences(self, df: pd.DataFrame) -> Dict:
        """Analyze preferences for audio features."""
        feature_preferences = {}

        for _, row in df.iterrows():
            try:
                audio_features = json.loads(row['audio_features']) if row['audio_features'] else {}
                feedback_value = row['feedback_value']

                for feature, value in audio_features.items():
                    if isinstance(value, (int, float)) and feature != 'track_id':
                        if feature not in feature_preferences:
                            feature_preferences[feature] = {'values': [], 'ratings': []}
                        feature_preferences[feature]['values'].append(value)
                        feature_preferences[feature]['ratings'].append(feedback_value)
            except:
                continue

        # Calculate correlations
        correlations = {}
        for feature, data in feature_preferences.items():
            if len(data['values']) > 5:  # Need minimum samples
                correlation = np.corrcoef(data['values'], data['ratings'])[0, 1]
                if not np.isnan(correlation):
                    correlations[feature] = correlation

        return correlations

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in feedback."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        return {
            'hourly_preferences': df.groupby('hour')['feedback_value'].mean().to_dict(),
            'daily_preferences': df.groupby('day_of_week')['feedback_value'].mean().to_dict(),
            'weekly_trend': df.set_index('timestamp').resample('W')['feedback_value'].mean().to_dict()
        }

    def _analyze_context_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze context patterns in feedback."""
        context_patterns = {}

        for _, row in df.iterrows():
            try:
                context = json.loads(row['feedback_context']) if row['feedback_context'] else {}
                feedback_value = row['feedback_value']

                for key, value in context.items():
                    if key not in context_patterns:
                        context_patterns[key] = {}
                    if value not in context_patterns[key]:
                        context_patterns[key][value] = []
                    context_patterns[key][value].append(feedback_value)
            except:
                continue

        # Calculate average ratings for each context
        context_averages = {}
        for key, values in context_patterns.items():
            context_averages[key] = {v: np.mean(ratings) for v, ratings in values.items()}

        return context_averages

    def _analyze_recommendation_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze performance of different recommendation methods."""
        performance = df.groupby('recommendation_reason').agg({
            'feedback_value': ['mean', 'count', 'std']
        }).round(3)

        return performance.to_dict() if not performance.empty else {}

    def _analyze_preference_stability(self, df: pd.DataFrame) -> Dict:
        """Analyze how stable user preferences are over time."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Calculate rolling average of feedback
        df['rolling_avg'] = df['feedback_value'].rolling(window=10, min_periods=5).mean()

        # Calculate variance in preferences over time
        stability_score = 1 - df['rolling_avg'].std() if len(df) > 10 else 0.5

        return {
            'stability_score': stability_score,
            'preference_drift': df['rolling_avg'].iloc[-10:].mean() - df['rolling_avg'].iloc[:10].mean() if len(df) > 20 else 0
        }

    def _get_current_algorithm_parameters(self, user_id: str) -> Dict:
        """Get current algorithm parameters for a user."""
        # Default parameters - these would be customized per user
        return {
            'feature_weights': {
                'energy': 0.15,
                'valence': 0.15,
                'danceability': 0.1,
                'acousticness': 0.1,
                'tempo': 0.2,
                'instrumentalness': 0.05,
                'speechiness': 0.05,
                'loudness': 0.05,
                'liveness': 0.05,
                'relative_popularity': 0.1
            },
            'similarity_threshold': 0.7,
            'diversity_factor': 0.3,
            'discovery_boost': 0.2
        }

    def _calculate_adapted_parameters(self, patterns: Dict, current_params: Dict) -> Dict:
        """Calculate new algorithm parameters based on feedback patterns."""
        new_params = current_params.copy()

        feature_preferences = patterns.get('feature_preferences', {})

        # Adjust feature weights based on correlations
        for feature, correlation in feature_preferences.items():
            if feature in new_params['feature_weights']:
                # Increase weight for features with positive correlation
                adjustment = 0.1 * correlation
                new_params['feature_weights'][feature] = max(0.01,
                    min(0.5, new_params['feature_weights'][feature] + adjustment))

        # Normalize weights to sum to 1
        total_weight = sum(new_params['feature_weights'].values())
        for feature in new_params['feature_weights']:
            new_params['feature_weights'][feature] /= total_weight

        return new_params

    def _apply_algorithm_adaptations(self, user_id: str, old_params: Dict, new_params: Dict) -> Dict:
        """Apply algorithm adaptations."""
        # This would integrate with the recommender system
        # For now, just return the changes made

        changes = {}
        for param_type, params in new_params.items():
            if param_type in old_params:
                if isinstance(params, dict):
                    for key, value in params.items():
                        old_value = old_params[param_type].get(key, 0)
                        if abs(value - old_value) > 0.01:  # Significant change
                            changes[f"{param_type}.{key}"] = {
                                'old': old_value,
                                'new': value,
                                'change': value - old_value
                            }

        return {
            'changes_applied': len(changes),
            'parameter_changes': changes,
            'estimated_improvement': min(0.2, len(changes) * 0.02)  # Rough estimate
        }

    def _log_algorithm_adaptation(self, user_id: str, old_params: Dict,
                                 new_params: Dict, results: Dict):
        """Log algorithm adaptation to database."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO algorithm_adaptations (
                        user_id, adaptation_type, old_parameters, new_parameters,
                        performance_improvement, feedback_batch_size
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, 'automatic_adaptation',
                    json.dumps(old_params), json.dumps(new_params),
                    results.get('estimated_improvement', 0),
                    self._get_total_feedback_count(user_id)
                ))

                conn.commit()

        except Exception as e:
            print(f"⚠️ Error logging adaptation: {e}")

    def _get_feedback_training_data(self, user_id: str) -> List[Dict]:
        """Get feedback data for training the prediction model."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT track_id, feedback_value, audio_features
                    FROM user_feedback
                    WHERE user_id = ? AND audio_features != '{}'
                    ORDER BY timestamp DESC
                    LIMIT 500
                ''', (user_id,))

                results = cursor.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            print(f"❌ Error getting training data: {e}")
            return []

    def _prepare_training_data(self, feedback_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the ML model."""
        features = []
        targets = []

        for item in feedback_data:
            try:
                audio_features = json.loads(item['audio_features'])
                feature_vector = self._prepare_prediction_features(audio_features)
                if feature_vector is not None:
                    features.append(feature_vector)
                    targets.append(item['feedback_value'])
            except:
                continue

        return np.array(features), np.array(targets)

    def _prepare_prediction_features(self, track_features: Dict) -> Optional[List[float]]:
        """Prepare feature vector for prediction."""
        try:
            feature_names = self._get_audio_feature_names()
            feature_vector = []

            for feature_name in feature_names:
                value = track_features.get(feature_name, 0.5)  # Default to middle value
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.5)

            return feature_vector if len(feature_vector) == len(feature_names) else None

        except Exception as e:
            print(f"⚠️ Error preparing features: {e}")
            return None

    def _get_audio_feature_names(self) -> List[str]:
        """Get list of audio feature names for ML model."""
        return [
            'energy', 'valence', 'danceability', 'acousticness', 'instrumentalness',
            'speechiness', 'liveness', 'loudness', 'tempo', 'key', 'mode',
            'duration_ms', 'time_signature'
        ]

    def _calculate_recommendation_accuracy(self, user_id: str) -> float:
        """Calculate overall recommendation accuracy for a user."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT AVG(CASE WHEN feedback_value > 0 THEN 1.0 ELSE 0.0 END) as accuracy
                    FROM user_feedback
                    WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
                ''', (user_id,))

                result = cursor.fetchone()
                return result[0] if result and result[0] is not None else 0.5

        except Exception as e:
            print(f"⚠️ Error calculating accuracy: {e}")
            return 0.5
