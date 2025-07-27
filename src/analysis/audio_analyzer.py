"""
Advanced Audio Analysis Module

Uses Essentia and librosa to extract comprehensive audio features from track previews.
Provides detailed technical analysis including:
- Spectral features (MFCCs, spectral centroid, rolloff, etc.)
- Rhythmic features (tempo, beats, rhythm patterns)
- Tonal features (key, chroma, harmonic analysis)
- Loudness and dynamics
- Timbre and texture analysis
"""

import os
import tempfile
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("⚠️  Essentia not available. Install with: pip install essentia-tensorflow")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  Librosa not available. Install with: pip install librosa")

from pydub import AudioSegment
import soundfile as sf


class AudioAnalyzer:
    """Advanced audio analysis using multiple audio processing libraries."""

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio analyzer.

        Args:
            sample_rate: Target sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.essentia_available = ESSENTIA_AVAILABLE
        self.librosa_available = LIBROSA_AVAILABLE

        if not (self.essentia_available or self.librosa_available):
            raise ImportError("Neither Essentia nor librosa is available. Please install at least one.")

        # Initialize Essentia algorithms if available
        if self.essentia_available:
            self._init_essentia_algorithms()

    def _init_essentia_algorithms(self):
        """Initialize Essentia algorithms for efficient feature extraction."""
        try:
            # Core algorithms
            self.windowing = es.Windowing(type='blackmanharris92')
            self.spectrum = es.Spectrum()
            self.spectral_peaks = es.SpectralPeaks()

            # Spectral features
            self.mfcc = es.MFCC(numberCoefficients=13)
            self.spectral_centroid = es.SpectralCentroid()
            self.spectral_rolloff = es.RollOff()
            self.spectral_flux = es.SpectralFlux()
            self.zero_crossing_rate = es.ZeroCrossingRate()

            # Rhythmic features
            self.rhythm_extractor = es.RhythmExtractor2013()
            self.beat_tracker = es.BeatTrackerMultiFeature()

            # Tonal features
            self.key_extractor = es.KeyExtractor()
            self.chromagram = es.ChromaCrossSimilarity()
            self.hpcp = es.HPCP()

            # Loudness and dynamics
            self.loudness = es.Loudness()
            self.dynamic_complexity = es.DynamicComplexity()

            print("✅ Essentia algorithms initialized")

        except Exception as e:
            print(f"⚠️  Error initializing Essentia algorithms: {e}")
            self.essentia_available = False

    def download_preview(self, preview_url: str) -> Optional[str]:
        """
        Download audio preview from Spotify.

        Args:
            preview_url: URL to 30-second preview

        Returns:
            Path to downloaded audio file or None if failed
        """
        if not preview_url:
            return None

        try:
            response = requests.get(preview_url, timeout=30)
            response.raise_for_status()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(response.content)
                return temp_file.name

        except Exception as e:
            print(f"❌ Error downloading preview: {e}")
            return None

    def load_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file and convert to the target sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate) or None if failed
        """
        try:
            # Try multiple loading methods
            audio_data = None
            sr = None

            # Method 1: librosa (preferred for analysis)
            if self.librosa_available:
                try:
                    audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                except Exception as e:
                    print(f"Librosa loading failed: {e}")

            # Method 2: soundfile
            if audio_data is None:
                try:
                    audio_data, sr = sf.read(file_path)
                    if len(audio_data.shape) > 1:  # Convert to mono
                        audio_data = np.mean(audio_data, axis=1)
                    if sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                except Exception as e:
                    print(f"Soundfile loading failed: {e}")

            # Method 3: pydub + conversion
            if audio_data is None:
                try:
                    audio = AudioSegment.from_file(file_path)
                    audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / (2**15)  # Normalize
                    sr = self.sample_rate
                except Exception as e:
                    print(f"Pydub loading failed: {e}")

            if audio_data is not None:
                return audio_data, sr
            else:
                print("❌ All audio loading methods failed")
                return None

        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None

    def extract_essentia_features(self, audio_data: np.ndarray) -> Dict:
        """
        Extract comprehensive audio features using Essentia.

        Args:
            audio_data: Audio time series

        Returns:
            Dictionary of extracted features
        """
        if not self.essentia_available:
            return {}

        try:
            features = {}

            # Convert to Essentia format
            audio_essentia = audio_data.astype(np.float32)

            # === SPECTRAL FEATURES ===

            # Frame-based analysis
            frames = []
            for frame in es.FrameGenerator(audio_essentia, frameSize=1024, hopSize=512):
                frames.append(frame)

            if frames:
                # MFCCs
                mfccs = []
                spectral_centroids = []
                spectral_rolloffs = []
                zero_crossings = []

                for frame in frames[:100]:  # Limit to avoid memory issues
                    windowed = self.windowing(frame)
                    spectrum = self.spectrum(windowed)

                    # MFCCs
                    mfcc_bands, mfcc_coeffs = self.mfcc(spectrum)
                    mfccs.append(mfcc_coeffs)

                    # Spectral features
                    spectral_centroids.append(self.spectral_centroid(spectrum))
                    spectral_rolloffs.append(self.spectral_rolloff(spectrum))
                    zero_crossings.append(self.zero_crossing_rate(frame))

                # Statistical summaries
                if mfccs:
                    mfccs = np.array(mfccs)
                    for i in range(min(13, mfccs.shape[1])):
                        features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[:, i]))
                        features[f'mfcc_{i}_std'] = float(np.std(mfccs[:, i]))

                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloffs))
                features['zero_crossing_rate_mean'] = float(np.mean(zero_crossings))

            # === RHYTHMIC FEATURES ===
            try:
                bpm, beats, beats_confidence, _, beats_intervals = self.rhythm_extractor(audio_essentia)
                features['tempo_essentia'] = float(bpm)
                features['beats_confidence'] = float(beats_confidence)
                features['num_beats'] = len(beats)
                if len(beats_intervals) > 0:
                    features['beat_interval_mean'] = float(np.mean(beats_intervals))
                    features['beat_interval_std'] = float(np.std(beats_intervals))
            except Exception as e:
                print(f"Rhythm extraction failed: {e}")

            # === TONAL FEATURES ===
            try:
                key, scale, strength = self.key_extractor(audio_essentia)
                features['key'] = key
                features['scale'] = scale
                features['key_strength'] = float(strength)
            except Exception as e:
                print(f"Key extraction failed: {e}")

            # === LOUDNESS AND DYNAMICS ===
            try:
                loudness_value = self.loudness(audio_essentia)
                features['loudness_essentia'] = float(loudness_value)

                dynamic_complexity_value = self.dynamic_complexity(audio_essentia)
                features['dynamic_complexity'] = float(dynamic_complexity_value)
            except Exception as e:
                print(f"Loudness analysis failed: {e}")

            return features

        except Exception as e:
            print(f"❌ Error extracting Essentia features: {e}")
            return {}

    def extract_librosa_features(self, audio_data: np.ndarray) -> Dict:
        """
        Extract audio features using librosa (backup method).

        Args:
            audio_data: Audio time series

        Returns:
            Dictionary of extracted features
        """
        if not self.librosa_available:
            return {}

        try:
            features = {}

            # === SPECTRAL FEATURES ===

            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

            # === RHYTHMIC FEATURES ===

            # Tempo and beats
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo_librosa'] = float(tempo)
            features['num_beats_librosa'] = len(beats)

            # === TONAL FEATURES ===

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))

            # === ENERGY AND DYNAMICS ===

            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            features['rms_energy_mean'] = float(np.mean(rms))
            features['rms_energy_std'] = float(np.std(rms))

            return features

        except Exception as e:
            print(f"❌ Error extracting librosa features: {e}")
            return {}

    def analyze_track(self, track_data: Dict) -> Optional[Dict]:
        """
        Perform comprehensive audio analysis on a track.

        Args:
            track_data: Track information including preview_url

        Returns:
            Dictionary of extracted audio features or None if failed
        """
        preview_url = track_data.get('preview_url')
        if not preview_url:
            print(f"❌ No preview URL for track: {track_data.get('name', 'Unknown')}")
            return None

        print(f"🎵 Analyzing: {track_data.get('name', 'Unknown')} by {track_data.get('artist', 'Unknown')}")

        # Download preview
        audio_file = self.download_preview(preview_url)
        if not audio_file:
            return None

        try:
            # Load audio
            audio_data, sr = self.load_audio(audio_file)
            if audio_data is None:
                return None

            # Extract features from both libraries
            features = {}

            # Essentia features (primary)
            if self.essentia_available:
                essentia_features = self.extract_essentia_features(audio_data)
                features.update(essentia_features)

            # Librosa features (backup/additional)
            if self.librosa_available:
                librosa_features = self.extract_librosa_features(audio_data)
                features.update(librosa_features)

            # Add basic metadata
            features['track_id'] = track_data.get('id')
            features['duration_analyzed'] = len(audio_data) / sr
            features['sample_rate'] = sr
            features['analysis_method'] = []
            if self.essentia_available:
                features['analysis_method'].append('essentia')
            if self.librosa_available:
                features['analysis_method'].append('librosa')
            features['analysis_method'] = ','.join(features['analysis_method'])

            return features

        except Exception as e:
            print(f"❌ Error analyzing track: {e}")
            return None

        finally:
            # Clean up temporary file
            try:
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)
            except Exception:
                pass

    def batch_analyze(self, tracks: List[Dict], max_tracks: int = None) -> List[Dict]:
        """
        Analyze multiple tracks in batch.

        Args:
            tracks: List of track dictionaries
            max_tracks: Maximum number of tracks to analyze

        Returns:
            List of feature dictionaries
        """
        if max_tracks:
            tracks = tracks[:max_tracks]

        results = []
        total = len(tracks)

        print(f"🎼 Starting batch analysis of {total} tracks...")

        for i, track in enumerate(tracks, 1):
            print(f"📊 Progress: {i}/{total}")

            features = self.analyze_track(track)
            if features:
                results.append(features)

            # Progress indicator
            if i % 10 == 0:
                print(f"✅ Analyzed {i}/{total} tracks ({len(results)} successful)")

        print(f"🎯 Batch analysis complete: {len(results)}/{total} tracks analyzed successfully")
        return results


def create_feature_similarity_matrix(features_df: pd.DataFrame) -> np.ndarray:
    """
    Create a similarity matrix from audio features for recommendation purposes.

    Args:
        features_df: DataFrame with audio features

    Returns:
        Similarity matrix
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity

        # Select numeric columns for similarity calculation
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        feature_matrix = features_df[numeric_columns].fillna(0)

        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(normalized_features)

        return similarity_matrix

    except Exception as e:
        print(f"❌ Error creating similarity matrix: {e}")
        return np.array([])
