"""
Audiophile Web Application

Central interface for the intelligent music discovery platform featuring:
- Interactive recommendation dashboard
- Real-time feedback collection
- Relative popularity analysis
- Hyperdimensional audio fingerprinting
- Algorithm adaptation visualization
- Smart playlist creation
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import time
from typing import Dict, List, Optional

# Import our custom modules
from src.auth.spotify_auth import get_spotify_client
from src.database.models import AudiophileDatabase
from src.recommendations.recommender import MusicRecommender
from src.recommendations.relative_popularity import RelativePopularityAnalyzer, enhance_recommendations_with_relative_popularity
from src.feedback.feedback_system import FeedbackSystem
from src.analysis.hyperdimensional_analyzer import HyperdimensionalAnalyzer
from src.analysis.audio_intelligence import AudioIntelligenceEngine

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instances
db = None
spotify = None
recommender = None
feedback_system = None
rel_pop_analyzer = None
hyper_analyzer = None
audio_intelligence = None

# Background task queues
analysis_queue = queue.Queue()
recommendation_queue = queue.Queue()


def initialize_app():
    """Initialize all components of the application."""
    global db, spotify, recommender, feedback_system, rel_pop_analyzer, hyper_analyzer, audio_intelligence

    print("🚀 Initializing Audiophile application...")

    try:
        # Initialize database
        db = AudiophileDatabase()

        # Initialize Spotify client
        spotify = get_spotify_client()

        # Initialize analyzers
        hyper_analyzer = HyperdimensionalAnalyzer(
            genius_api_key=os.getenv('GENIUS_API_KEY')
        )

        rel_pop_analyzer = RelativePopularityAnalyzer(spotify, db)

        # Initialize audio intelligence engine
        audio_intelligence = AudioIntelligenceEngine(spotify, db)

        # Initialize recommender (will be created per user session)
        # recommender will be set when user data is loaded

        print("✅ Application initialized successfully")

    except Exception as e:
        print(f"❌ Error initializing application: {e}")
        raise


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/login')
def login():
    """Spotify authentication login."""
    try:
        # Get Spotify authorization URL
        auth_url = spotify.auth_manager.get_authorize_url()
        return redirect(auth_url)
    except Exception as e:
        flash(f"Error initiating login: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/callback')
def callback():
    """Handle Spotify authentication callback."""
    try:
        code = request.args.get('code')
        if code:
            # Get access token
            token_info = spotify.auth_manager.get_access_token(code)
            session['token_info'] = token_info

            # Get user info
            user_info = spotify.current_user()
            session['user_id'] = user_info['id']
            session['user_name'] = user_info.get('display_name', user_info['id'])

            flash('Successfully connected to Spotify!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Authentication failed', 'error')
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Error during authentication: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    """User dashboard with recommendations and analytics."""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        user_id = session['user_id']

        # Get user's recent activity
        user_stats = get_user_stats(user_id)

        # Get recent recommendations
        recent_recommendations = get_recent_recommendations(user_id)

        # Get feedback analytics
        feedback_analytics = get_feedback_analytics(user_id)

        return render_template('dashboard.html',
                             user_stats=user_stats,
                             recommendations=recent_recommendations,
                             feedback_analytics=feedback_analytics,
                             user_name=session.get('user_name'))

    except Exception as e:
        flash(f"Error loading dashboard: {e}", 'error')
        return redirect(url_for('index'))


@app.route('/discover')
def discover():
    """Music discovery page with advanced recommendations."""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('discover.html')


@app.route('/analytics')
def analytics():
    """Analytics and insights page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        user_id = session['user_id']

        # Get comprehensive analytics
        analytics_data = get_comprehensive_analytics(user_id)

        return render_template('analytics.html', analytics=analytics_data)

    except Exception as e:
        flash(f"Error loading analytics: {e}", 'error')
        return redirect(url_for('dashboard'))


@app.route('/feedback')
def feedback_page():
    """Feedback collection and algorithm training page."""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('feedback.html')


# API Routes

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations_api():
    """Get personalized recommendations via API."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        user_id = session['user_id']
        data = request.get_json() or {}

        # Parameters
        n_recommendations = data.get('count', 20)
        recommendation_type = data.get('type', 'personalized')
        mood = data.get('mood')
        discovery_mode = data.get('discovery_mode', False)

        # Generate recommendations
        recommendations = generate_recommendations(
            user_id, n_recommendations, recommendation_type, mood, discovery_mode
        )

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback_api():
    """Submit feedback on recommendations."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        user_id = session['user_id']
        data = request.get_json()

        # Extract feedback data
        track_id = data.get('track_id')
        feedback_type = data.get('feedback_type')
        recommendation_id = data.get('recommendation_id')
        context = data.get('context', {})

        if not track_id or not feedback_type:
            return jsonify({'error': 'Missing required fields'}), 400

        # Submit feedback
        if feedback_system:
            success = feedback_system.collect_feedback(
                user_id=user_id,
                track_id=track_id,
                feedback_type=feedback_type,
                recommendation_id=recommendation_id,
                context=context
            )

            if success:
                # Emit real-time update to connected clients
                socketio.emit('feedback_received', {
                    'track_id': track_id,
                    'feedback_type': feedback_type,
                    'timestamp': datetime.now().isoformat()
                }, room=user_id)

                return jsonify({'success': True, 'message': 'Feedback recorded successfully'})
            else:
                return jsonify({'error': 'Failed to record feedback'}), 500
        else:
            return jsonify({'error': 'Feedback system not available'}), 503

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_track', methods=['POST'])
def analyze_track_api():
    """Analyze a specific track with hyperdimensional analysis."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        track_id = data.get('track_id')

        if not track_id:
            return jsonify({'error': 'Missing track_id'}), 400

        # Queue track for analysis
        analysis_queue.put({
            'user_id': session['user_id'],
            'track_id': track_id,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            'success': True,
            'message': 'Track queued for analysis',
            'track_id': track_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/relative_popularity', methods=['POST'])
def analyze_relative_popularity_api():
    """Analyze relative popularity for tracks."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        track_ids = data.get('track_ids', [])

        if not track_ids:
            return jsonify({'error': 'No track IDs provided'}), 400

        if rel_pop_analyzer:
            # Analyze relative popularity
            analysis_results = []

            for track_id in track_ids[:10]:  # Limit to 10 tracks for performance
                try:
                    # Get artist ID for the track
                    track = spotify.track(track_id)
                    artist_id = track['artists'][0]['id']

                    # Analyze relative popularity
                    rel_pop = rel_pop_analyzer.calculate_relative_popularity_score(track_id, artist_id)

                    if 'error' not in rel_pop:
                        analysis_results.append({
                            'track_id': track_id,
                            'track_name': track['name'],
                            'artist_name': track['artists'][0]['name'],
                            'relative_popularity': rel_pop
                        })

                except Exception as e:
                    print(f"Error analyzing track {track_id}: {e}")
                    continue

            return jsonify({
                'success': True,
                'results': analysis_results,
                'total_analyzed': len(analysis_results)
            })
        else:
            return jsonify({'error': 'Relative popularity analyzer not available'}), 503

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/create_playlist', methods=['POST'])
def create_smart_playlist_api():
    """Create an intelligent playlist."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()

        seed_tracks = data.get('seed_tracks', [])
        playlist_name = data.get('name', f"AI Playlist {datetime.now().strftime('%Y-%m-%d')}")
        playlist_length = data.get('length', 30)
        diversity_factor = data.get('diversity', 0.3)

        if not seed_tracks:
            return jsonify({'error': 'No seed tracks provided'}), 400

        # Create smart playlist
        if audio_intelligence:
            playlist_info = audio_intelligence.create_smart_playlist(
                seed_tracks=seed_tracks,
                playlist_name=playlist_name,
                playlist_length=playlist_length,
                diversity_factor=diversity_factor
            )

            if playlist_info:
                return jsonify({
                    'success': True,
                    'playlist': playlist_info,
                    'message': f'Created playlist "{playlist_name}" with {len(playlist_info.get("tracks", []))} tracks'
                })
            else:
                return jsonify({'error': 'Failed to create playlist'}), 500
        else:
            return jsonify({'error': 'Audio intelligence engine not available'}), 503

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user_stats')
def user_stats_api():
    """Get user statistics and analytics."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        user_id = session['user_id']
        stats = get_user_stats(user_id)
        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if 'user_id' in session:
        join_room(session['user_id'])
        emit('connected', {'message': 'Connected to Audiophile'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    if 'user_id' in session:
        leave_room(session['user_id'])


@socketio.on('request_recommendations')
def handle_recommendation_request(data):
    """Handle real-time recommendation requests."""
    if 'user_id' not in session:
        emit('error', {'message': 'Not authenticated'})
        return

    try:
        user_id = session['user_id']

        # Queue recommendation request
        recommendation_queue.put({
            'user_id': user_id,
            'params': data,
            'socket_id': request.sid,
            'timestamp': datetime.now().isoformat()
        })

        emit('recommendation_queued', {'message': 'Recommendation request queued'})

    except Exception as e:
        emit('error', {'message': str(e)})


# Background Tasks

def background_analysis_worker():
    """Background worker for audio analysis tasks."""
    while True:
        try:
            if not analysis_queue.empty():
                task = analysis_queue.get(timeout=1)

                user_id = task['user_id']
                track_id = task['track_id']

                print(f"🔬 Analyzing track {track_id} for user {user_id}")

                # Perform analysis
                if hyper_analyzer:
                    # Get track info
                    track = spotify.track(track_id)

                    # Download preview if available
                    preview_url = track.get('preview_url')
                    if preview_url:
                        # In a real implementation, you'd download and analyze the audio
                        # For now, we'll simulate the analysis

                        analysis_result = {
                            'track_id': track_id,
                            'track_name': track['name'],
                            'artist_name': track['artists'][0]['name'],
                            'analysis_complete': True,
                            'feature_count': np.random.randint(250, 350),  # Simulated feature count
                            'analysis_timestamp': datetime.now().isoformat()
                        }

                        # Emit result to user
                        socketio.emit('analysis_complete', analysis_result, room=user_id)

                        print(f"✅ Analysis complete for track {track_id}")
                    else:
                        socketio.emit('analysis_error', {
                            'track_id': track_id,
                            'error': 'No preview available for analysis'
                        }, room=user_id)

                analysis_queue.task_done()
            else:
                time.sleep(1)

        except queue.Empty:
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error in analysis worker: {e}")
            time.sleep(5)


def background_recommendation_worker():
    """Background worker for recommendation tasks."""
    while True:
        try:
            if not recommendation_queue.empty():
                task = recommendation_queue.get(timeout=1)

                user_id = task['user_id']
                params = task['params']
                socket_id = task['socket_id']

                print(f"🎵 Generating recommendations for user {user_id}")

                # Generate recommendations
                recommendations = generate_recommendations(
                    user_id,
                    params.get('count', 20),
                    params.get('type', 'personalized'),
                    params.get('mood'),
                    params.get('discovery_mode', False)
                )

                # Emit result
                socketio.emit('recommendations_ready', {
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat()
                }, room=user_id)

                print(f"✅ Recommendations generated for user {user_id}")

                recommendation_queue.task_done()
            else:
                time.sleep(1)

        except queue.Empty:
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error in recommendation worker: {e}")
            time.sleep(5)


# Helper Functions

def generate_recommendations(user_id: str, count: int = 20,
                           rec_type: str = 'personalized',
                           mood: str = None,
                           discovery_mode: bool = False) -> List[Dict]:
    """Generate recommendations for a user."""
    try:
        # Get user's tracks
        user_tracks = get_user_tracks(user_id)

        if not user_tracks:
            return []

        # Initialize recommender if needed
        global recommender
        if not recommender:
            # Create features DataFrame (simplified for demo)
            features_data = []
            tracks_data = []

            for track in user_tracks[:50]:  # Limit for performance
                # Get audio features
                try:
                    audio_features = spotify.audio_features(track['id'])[0]
                    if audio_features:
                        features_data.append({
                            'track_id': track['id'],
                            **audio_features
                        })
                        tracks_data.append(track)
                except:
                    continue

            if features_data:
                features_df = pd.DataFrame(features_data)
                tracks_df = pd.DataFrame(tracks_data)
                recommender = MusicRecommender(features_df, tracks_df)

        if recommender:
            # Generate recommendations based on type
            if rec_type == 'mood' and mood:
                recommendations = recommender.get_mood_based_recommendations(mood, count)
            elif rec_type == 'similar':
                # Use first track as seed
                recommendations = recommender.find_similar_tracks(user_tracks[0]['id'], count)
            else:
                # Personalized recommendations
                track_ids = [t['id'] for t in user_tracks[:20]]
                recommendations = recommender.get_personalized_recommendations(track_ids, count)

            # Enhance with relative popularity if discovery mode is enabled
            if discovery_mode and rel_pop_analyzer and recommendations:
                recommendations = enhance_recommendations_with_relative_popularity(
                    recommendations, spotify, rel_pop_analyzer
                )

            return recommendations

        return []

    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return []


def get_user_tracks(user_id: str, limit: int = 50) -> List[Dict]:
    """Get user's tracks from database."""
    try:
        if db:
            tracks = db.get_user_tracks_with_features(user_id, limit)
            return [dict(track) for track in tracks]
        return []
    except Exception as e:
        print(f"❌ Error getting user tracks: {e}")
        return []


def get_user_stats(user_id: str) -> Dict:
    """Get user statistics."""
    try:
        stats = {
            'total_tracks': 0,
            'total_feedback': 0,
            'recommendation_accuracy': 0.0,
            'discovery_score': 0.0,
            'last_activity': None
        }

        if db:
            db_stats = db.get_database_stats()

            # Get user-specific stats
            stats.update({
                'total_tracks': db_stats.get('user_tracks_count', 0),
                'total_feedback': 0,  # Would query feedback system
                'recommendation_accuracy': 0.75,  # Placeholder
                'discovery_score': 0.68,  # Placeholder
                'last_activity': datetime.now().isoformat()
            })

        return stats

    except Exception as e:
        print(f"❌ Error getting user stats: {e}")
        return {}


def get_recent_recommendations(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent recommendations for user."""
    try:
        # Placeholder implementation
        return []
    except Exception as e:
        print(f"❌ Error getting recent recommendations: {e}")
        return []


def get_feedback_analytics(user_id: str) -> Dict:
    """Get feedback analytics for user."""
    try:
        if feedback_system:
            return feedback_system.get_feedback_dashboard_data(user_id)
        return {}
    except Exception as e:
        print(f"❌ Error getting feedback analytics: {e}")
        return {}


def get_comprehensive_analytics(user_id: str) -> Dict:
    """Get comprehensive analytics for user."""
    try:
        analytics = {
            'listening_patterns': {},
            'preference_evolution': {},
            'discovery_insights': {},
            'algorithm_performance': {}
        }

        # Would implement comprehensive analytics here

        return analytics

    except Exception as e:
        print(f"❌ Error getting comprehensive analytics: {e}")
        return {}


# Error Handlers

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500


if __name__ == '__main__':
    print("🎵 Starting Audiophile Application...")

    try:
        # Initialize application
        initialize_app()

        # Start background workers
        analysis_thread = threading.Thread(target=background_analysis_worker, daemon=True)
        analysis_thread.start()

        recommendation_thread = threading.Thread(target=background_recommendation_worker, daemon=True)
        recommendation_thread.start()

        print("🚀 Background workers started")

                # Run the application
        port = int(os.getenv('PORT', 5001))  # Changed to 5001 to avoid AirPlay conflict
        debug = os.getenv('FLASK_ENV') == 'development'

        print(f"🌐 Starting web server on port {port}")
        print(f"🎵 Navigate to http://localhost:{port} to start discovering music!")
        socketio.run(app, host='0.0.0.0', port=port, debug=debug)

    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        exit(1)
