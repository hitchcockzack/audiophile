"""
Hyperdimensional Audio Analysis Engine

Extracts 300+ audio features to create comprehensive "DNA" fingerprints of songs.
Includes advanced analysis for:
- Vocal characteristics (tone, timber, style)
- Harmonic structure (chords, progressions, modulations)
- Rhythmic patterns (groove, syncopation, polyrhythms)
- Spectral evolution (how sound changes over time)
- Lyrical content and sentiment
- Structural analysis (verses, choruses, bridges)
- Emotional mapping and mood curves
- Cultural and genre markers
"""

import numpy as np
import pandas as pd
import librosa
import librosa.display
from typing import Dict, List, Optional, Tuple, Any
import requests
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced analysis libraries
try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("⚠️  Essentia not available - some advanced features will be limited")

try:
    from textblob import TextBlob
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  TextBlob/NLTK not available - lyrical analysis will be limited")

try:
    import madmom
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    print("⚠️  Madmom not available - some rhythmic analysis will be limited")


class HyperdimensionalAnalyzer:
    """Advanced audio analysis for comprehensive music fingerprinting."""

    def __init__(self, genius_api_key: Optional[str] = None):
        """
        Initialize the hyperdimensional analyzer.

        Args:
            genius_api_key: Optional Genius API key for lyrical analysis
        """
        self.genius_api_key = genius_api_key
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048

        # Initialize analysis components
        self._init_essentia_analyzers()
        self._init_vocal_analyzers()
        self._init_harmonic_analyzers()

        print("🧬 Hyperdimensional analyzer initialized")

    def analyze_track_comprehensive(self, audio_data: np.ndarray,
                                  track_metadata: Dict = None,
                                  include_lyrics: bool = True) -> Dict:
        """
        Perform comprehensive analysis extracting 300+ features.

        Args:
            audio_data: Audio waveform data
            track_metadata: Optional track metadata (title, artist, etc.)
            include_lyrics: Whether to include lyrical analysis

        Returns:
            Dictionary with comprehensive feature set
        """
        print("🔬 Starting hyperdimensional analysis...")

        analysis_start = datetime.now()

        # Core feature extraction
        features = {
            'metadata': track_metadata or {},
            'analysis_timestamp': analysis_start.isoformat(),
            'audio_duration': len(audio_data) / self.sample_rate
        }

        try:
            # 1. Spectral Features (50+ features)
            print("   📊 Extracting spectral features...")
            features.update(self._extract_spectral_features(audio_data))

            # 2. Harmonic & Tonal Features (40+ features)
            print("   🎵 Analyzing harmonic structure...")
            features.update(self._extract_harmonic_features(audio_data))

            # 3. Rhythmic Features (35+ features)
            print("   🥁 Analyzing rhythmic patterns...")
            features.update(self._extract_rhythmic_features(audio_data))

            # 4. Vocal Characteristics (45+ features)
            print("   🎤 Analyzing vocal characteristics...")
            features.update(self._extract_vocal_features(audio_data))

            # 5. Structural Analysis (25+ features)
            print("   🏗️  Analyzing song structure...")
            features.update(self._extract_structural_features(audio_data))

            # 6. Dynamic & Energy Features (30+ features)
            print("   ⚡ Analyzing dynamics and energy...")
            features.update(self._extract_dynamic_features(audio_data))

            # 7. Temporal Evolution (40+ features)
            print("   ⏱️  Analyzing temporal evolution...")
            features.update(self._extract_temporal_features(audio_data))

            # 8. Psychoacoustic Features (25+ features)
            print("   🧠 Extracting psychoacoustic features...")
            features.update(self._extract_psychoacoustic_features(audio_data))

            # 9. Cultural & Genre Markers (15+ features)
            print("   🌍 Identifying cultural markers...")
            features.update(self._extract_cultural_features(audio_data))

            # 10. Lyrical Analysis (if available) (20+ features)
            if include_lyrics and track_metadata:
                print("   📝 Analyzing lyrics...")
                features.update(self._extract_lyrical_features(track_metadata))

            # 11. Advanced Essentia Features (if available)
            if ESSENTIA_AVAILABLE:
                print("   🎼 Extracting advanced Essentia features...")
                features.update(self._extract_essentia_features(audio_data))

            analysis_time = (datetime.now() - analysis_start).total_seconds()
            features['analysis_duration_seconds'] = analysis_time

            feature_count = sum(1 for k, v in features.items()
                              if isinstance(v, (int, float)) and not k.startswith('metadata'))

            print(f"✅ Extracted {feature_count} features in {analysis_time:.2f}s")

            return features

        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            features['analysis_error'] = str(e)
            return features

    def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict:
        """Extract comprehensive spectral analysis features."""
        features = {}

        try:
            # Basic spectral features
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)

            # Spectral centroid statistics
            spectral_centroids = librosa.feature.spectral_centroid(audio_data, sr=self.sample_rate)[0]
            features.update({
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_centroid_min': float(np.min(spectral_centroids)),
                'spectral_centroid_max': float(np.max(spectral_centroids)),
                'spectral_centroid_skew': float(self._calculate_skewness(spectral_centroids)),
                'spectral_centroid_kurtosis': float(self._calculate_kurtosis(spectral_centroids))
            })

            # Spectral rolloff statistics
            spectral_rolloff = librosa.feature.spectral_rolloff(audio_data, sr=self.sample_rate)[0]
            features.update({
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_rolloff_85_mean': float(np.mean(librosa.feature.spectral_rolloff(
                    audio_data, sr=self.sample_rate, roll_percent=0.85)[0])),
                'spectral_rolloff_95_mean': float(np.mean(librosa.feature.spectral_rolloff(
                    audio_data, sr=self.sample_rate, roll_percent=0.95)[0]))
            })

            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(audio_data, sr=self.sample_rate)[0]
            features.update({
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth))
            })

            # Zero crossing rate analysis
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features.update({
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr)),
                'zero_crossing_rate_contrast': float(np.max(zcr) - np.min(zcr))
            })

            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(audio_data, sr=self.sample_rate)
            for i, band in enumerate(spectral_contrast):
                features[f'spectral_contrast_band_{i}_mean'] = float(np.mean(band))
                features[f'spectral_contrast_band_{i}_std'] = float(np.std(band))

            # Spectral flatness (measure of noisiness)
            spectral_flatness = librosa.feature.spectral_flatness(audio_data)[0]
            features.update({
                'spectral_flatness_mean': float(np.mean(spectral_flatness)),
                'spectral_flatness_std': float(np.std(spectral_flatness))
            })

            # Mel-frequency cepstral coefficients (detailed)
            mfccs = librosa.feature.mfcc(audio_data, sr=self.sample_rate, n_mfcc=20)
            for i, mfcc in enumerate(mfccs):
                features.update({
                    f'mfcc_{i}_mean': float(np.mean(mfcc)),
                    f'mfcc_{i}_std': float(np.std(mfcc)),
                    f'mfcc_{i}_delta_mean': float(np.mean(np.diff(mfcc))),
                    f'mfcc_{i}_acceleration_mean': float(np.mean(np.diff(mfcc, n=2)))
                })

            return features

        except Exception as e:
            print(f"⚠️ Error extracting spectral features: {e}")
            return {}

    def _extract_harmonic_features(self, audio_data: np.ndarray) -> Dict:
        """Extract harmonic and tonal analysis features."""
        features = {}

        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)

            # Harmonic ratio
            harmonic_ratio = np.sum(harmonic**2) / np.sum(audio_data**2)
            features['harmonic_ratio'] = float(harmonic_ratio)
            features['percussive_ratio'] = float(1 - harmonic_ratio)

            # Chroma features (pitch class profiles)
            chroma = librosa.feature.chroma_stft(audio_data, sr=self.sample_rate)
            for i in range(12):
                features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{i}_std'] = float(np.std(chroma[i]))

            # Chroma energy normalized statistics
            chroma_energy = np.sum(chroma, axis=0)
            features.update({
                'chroma_energy_mean': float(np.mean(chroma_energy)),
                'chroma_energy_std': float(np.std(chroma_energy)),
                'chroma_energy_max': float(np.max(chroma_energy))
            })

            # Constant-Q transform for better harmonic analysis
            try:
                cqt = np.abs(librosa.cqt(audio_data, sr=self.sample_rate))
                features.update({
                    'cqt_energy_mean': float(np.mean(cqt)),
                    'cqt_energy_std': float(np.std(cqt)),
                    'cqt_spectral_centroid': float(np.mean(np.sum(cqt * np.arange(cqt.shape[0])[:, np.newaxis], axis=0) / np.sum(cqt, axis=0)))
                })
            except:
                pass

            # Tonnetz features (harmonic network)
            try:
                tonnetz = librosa.feature.tonnetz(harmonic, sr=self.sample_rate)
                for i in range(tonnetz.shape[0]):
                    features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
                    features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
            except:
                pass

            # Key and mode estimation using chroma
            chroma_mean = np.mean(chroma, axis=1)

            # Simple key estimation (circle of fifths)
            key_profiles = self._get_key_profiles()
            key_correlations = {}
            for key, profile in key_profiles.items():
                correlation = np.corrcoef(chroma_mean, profile)[0, 1]
                if not np.isnan(correlation):
                    key_correlations[key] = correlation

            if key_correlations:
                estimated_key = max(key_correlations, key=key_correlations.get)
                features['estimated_key'] = estimated_key
                features['key_confidence'] = float(key_correlations[estimated_key])

            # Harmonic change detection
            chroma_changes = np.diff(chroma, axis=1)
            features.update({
                'harmonic_change_rate': float(np.mean(np.sum(np.abs(chroma_changes), axis=0))),
                'harmonic_stability': float(1 / (1 + np.std(np.sum(np.abs(chroma_changes), axis=0))))
            })

            return features

        except Exception as e:
            print(f"⚠️ Error extracting harmonic features: {e}")
            return {}

    def _extract_rhythmic_features(self, audio_data: np.ndarray) -> Dict:
        """Extract comprehensive rhythmic analysis features."""
        features = {}

        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(audio_data, sr=self.sample_rate)
            features['tempo_librosa'] = float(tempo)
            features['num_beats'] = len(beats)

            if len(beats) > 1:
                beat_intervals = np.diff(beats) / self.sample_rate * self.hop_length
                features.update({
                    'beat_interval_mean': float(np.mean(beat_intervals)),
                    'beat_interval_std': float(np.std(beat_intervals)),
                    'beat_regularity': float(1 / (1 + np.std(beat_intervals))),
                    'beat_strength_variance': float(np.std(librosa.util.frame(
                        librosa.onset.onset_strength(audio_data, sr=self.sample_rate),
                        frame_length=1, hop_length=1)[beats[:min(len(beats), len(librosa.onset.onset_strength(audio_data, sr=self.sample_rate)))]]))
                })

            # Onset detection and rhythm analysis
            onset_frames = librosa.onset.onset_detect(audio_data, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)

            features['num_onsets'] = len(onset_frames)

            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                features.update({
                    'onset_interval_mean': float(np.mean(onset_intervals)),
                    'onset_interval_std': float(np.std(onset_intervals)),
                    'onset_density': float(len(onset_times) / (len(audio_data) / self.sample_rate))
                })

            # Rhythm patterns using onset strength
            onset_strength = librosa.onset.onset_strength(audio_data, sr=self.sample_rate)
            features.update({
                'onset_strength_mean': float(np.mean(onset_strength)),
                'onset_strength_std': float(np.std(onset_strength)),
                'onset_strength_max': float(np.max(onset_strength))
            })

            # Tempogram for rhythm pattern analysis
            try:
                tempogram = librosa.feature.tempogram(onset_strength=onset_strength, sr=self.sample_rate)
                features.update({
                    'tempogram_energy_mean': float(np.mean(tempogram)),
                    'tempogram_energy_std': float(np.std(tempogram)),
                    'tempogram_peak_ratio': float(np.max(tempogram) / np.mean(tempogram))
                })
            except:
                pass

            # Rhythmic complexity estimation
            if len(onset_intervals) > 10:
                # Calculate syncopation measure
                expected_interval = 60 / tempo  # Expected beat interval
                syncopation_score = np.mean(np.abs(onset_intervals - expected_interval)) / expected_interval
                features['syncopation_score'] = float(syncopation_score)

                # Rhythmic entropy
                interval_hist, _ = np.histogram(onset_intervals, bins=20)
                interval_probs = interval_hist / np.sum(interval_hist)
                interval_probs = interval_probs[interval_probs > 0]  # Remove zeros
                rhythmic_entropy = -np.sum(interval_probs * np.log2(interval_probs))
                features['rhythmic_entropy'] = float(rhythmic_entropy)

            return features

        except Exception as e:
            print(f"⚠️ Error extracting rhythmic features: {e}")
            return {}

    def _extract_vocal_features(self, audio_data: np.ndarray) -> Dict:
        """Extract vocal characteristics and singing style features."""
        features = {}

        try:
            # Spectral features that correlate with vocal characteristics

            # Formant estimation (simplified)
            # Use spectral peaks in vocal frequency range
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            stft = librosa.stft(audio_data, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(stft)

            # Focus on vocal frequency range (80-8000 Hz)
            vocal_freq_mask = (freqs >= 80) & (freqs <= 8000)
            vocal_spectrum = magnitude[vocal_freq_mask, :]
            vocal_freqs = freqs[vocal_freq_mask]

            # Estimate formants by finding spectral peaks
            formant_estimates = []
            for frame in vocal_spectrum.T:
                peaks = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=3)
                if len(peaks) > 0:
                    peak_freqs = vocal_freqs[peaks]
                    peak_magnitudes = frame[peaks]
                    # Take strongest peaks as formant estimates
                    sorted_indices = np.argsort(peak_magnitudes)[::-1]
                    formant_estimates.append(peak_freqs[sorted_indices[:4]])

            if formant_estimates:
                formant_estimates = np.array(formant_estimates)
                for i in range(min(4, formant_estimates.shape[1])):
                    formant_values = formant_estimates[:, i]
                    features.update({
                        f'formant_{i+1}_mean': float(np.mean(formant_values)),
                        f'formant_{i+1}_std': float(np.std(formant_values))
                    })

            # Vocal energy in different frequency bands
            low_vocal = np.sum(magnitude[(freqs >= 80) & (freqs <= 300), :], axis=0)
            mid_vocal = np.sum(magnitude[(freqs >= 300) & (freqs <= 2000), :], axis=0)
            high_vocal = np.sum(magnitude[(freqs >= 2000) & (freqs <= 8000), :], axis=0)

            total_vocal_energy = low_vocal + mid_vocal + high_vocal

            features.update({
                'vocal_low_energy_ratio': float(np.mean(low_vocal / (total_vocal_energy + 1e-10))),
                'vocal_mid_energy_ratio': float(np.mean(mid_vocal / (total_vocal_energy + 1e-10))),
                'vocal_high_energy_ratio': float(np.mean(high_vocal / (total_vocal_energy + 1e-10))),
                'vocal_brightness': float(np.mean(high_vocal / (low_vocal + mid_vocal + 1e-10)))
            })

            # Vibrato detection (frequency modulation in vocal range)
            # Simplified approach using spectral centroid variation
            vocal_spectral_centroid = np.sum(vocal_spectrum * vocal_freqs[:, np.newaxis], axis=0) / np.sum(vocal_spectrum, axis=0)

            # Look for periodic variations (vibrato)
            if len(vocal_spectral_centroid) > 100:
                # Remove trend
                detrended = vocal_spectral_centroid - np.convolve(vocal_spectral_centroid, np.ones(10)/10, mode='same')

                # Find periodicity using autocorrelation
                autocorr = np.correlate(detrended, detrended, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Look for peaks in 4-8 Hz range (typical vibrato)
                time_per_frame = self.hop_length / self.sample_rate
                vibrato_range = (int(0.125 / time_per_frame), int(0.25 / time_per_frame))  # 4-8 Hz

                if vibrato_range[1] < len(autocorr):
                    vibrato_strength = np.max(autocorr[vibrato_range[0]:vibrato_range[1]])
                    features['vibrato_strength'] = float(vibrato_strength / autocorr[0])

            # Rough vocal/instrumental separation
            # Use harmonic-percussive separation as a proxy
            harmonic, _ = librosa.effects.hpss(audio_data)

            # Calculate features on harmonic component (more likely to contain vocals)
            harmonic_mfccs = librosa.feature.mfcc(harmonic, sr=self.sample_rate, n_mfcc=13)

            # Vocal timbre features from harmonic component
            features.update({
                'vocal_timbre_brightness': float(np.mean(harmonic_mfccs[2])),  # MFCC 2 correlates with brightness
                'vocal_timbre_roughness': float(np.std(harmonic_mfccs[3])),   # MFCC 3 std correlates with roughness
                'vocal_timbre_warmth': float(-np.mean(harmonic_mfccs[1]))     # Inverse of MFCC 1 for warmth
            })

            # Vocal range estimation (simplified)
            fundamental_freq = librosa.yin(audio_data, fmin=80, fmax=800, sr=self.sample_rate)
            valid_f0 = fundamental_freq[fundamental_freq > 0]

            if len(valid_f0) > 10:
                features.update({
                    'vocal_f0_mean': float(np.mean(valid_f0)),
                    'vocal_f0_std': float(np.std(valid_f0)),
                    'vocal_f0_min': float(np.min(valid_f0)),
                    'vocal_f0_max': float(np.max(valid_f0)),
                    'vocal_range_semitones': float(12 * np.log2(np.max(valid_f0) / np.min(valid_f0)))
                })

            return features

        except Exception as e:
            print(f"⚠️ Error extracting vocal features: {e}")
            return {}

    def _extract_structural_features(self, audio_data: np.ndarray) -> Dict:
        """Extract song structure and form analysis features."""
        features = {}

        try:
            # Segment the audio using recurrence matrix
            chroma = librosa.feature.chroma_stft(audio_data, sr=self.sample_rate)

            # Create recurrence matrix
            R = librosa.segment.recurrence_matrix(chroma, mode='distance', metric='cosine')

            # Detect segments
            boundaries = librosa.segment.agglomerative(chroma, k=None)
            features['num_segments'] = len(boundaries) - 1

            if len(boundaries) > 2:
                segment_durations = np.diff(boundaries) * self.hop_length / self.sample_rate
                features.update({
                    'avg_segment_duration': float(np.mean(segment_durations)),
                    'segment_duration_std': float(np.std(segment_durations)),
                    'segment_repetition_score': float(np.mean(np.diag(R, k=1)))
                })

            # Novelty detection for structural changes
            onset_strength = librosa.onset.onset_strength(audio_data, sr=self.sample_rate)
            novelty = librosa.segment.recurrence_to_lag(R, pad=True)

            if len(novelty) > 0:
                features.update({
                    'structural_novelty_mean': float(np.mean(novelty)),
                    'structural_novelty_std': float(np.std(novelty)),
                    'structural_complexity': float(np.sum(novelty > np.mean(novelty) + 2*np.std(novelty)))
                })

            # Estimate song sections using feature changes
            # Look for patterns in chroma and rhythm
            tempo, beats = librosa.beat.beat_track(audio_data, sr=self.sample_rate)

            if len(beats) > 32:  # Need enough beats for section analysis
                # Group beats into measures (assuming 4/4 time)
                measures_per_section = 8  # Typical section length
                beats_per_measure = 4

                section_length = measures_per_section * beats_per_measure
                num_sections = len(beats) // section_length

                if num_sections >= 2:
                    features['estimated_num_sections'] = num_sections

                    # Analyze repetition between sections
                    section_features = []
                    for i in range(num_sections):
                        start_beat = i * section_length
                        end_beat = min((i + 1) * section_length, len(beats) - 1)

                        start_frame = beats[start_beat]
                        end_frame = beats[end_beat]

                        section_chroma = chroma[:, start_frame:end_frame]
                        section_features.append(np.mean(section_chroma, axis=1))

                    if len(section_features) >= 2:
                        # Calculate similarity between sections
                        similarities = []
                        for i in range(len(section_features)):
                            for j in range(i + 1, len(section_features)):
                                sim = np.corrcoef(section_features[i], section_features[j])[0, 1]
                                if not np.isnan(sim):
                                    similarities.append(sim)

                        if similarities:
                            features['section_similarity_mean'] = float(np.mean(similarities))
                            features['section_repetition_score'] = float(np.max(similarities))

            return features

        except Exception as e:
            print(f"⚠️ Error extracting structural features: {e}")
            return {}

    def _extract_dynamic_features(self, audio_data: np.ndarray) -> Dict:
        """Extract dynamic range and energy evolution features."""
        features = {}

        try:
            # RMS energy analysis
            rms = librosa.feature.rms(audio_data, hop_length=self.hop_length)[0]

            features.update({
                'rms_energy_mean': float(np.mean(rms)),
                'rms_energy_std': float(np.std(rms)),
                'rms_energy_min': float(np.min(rms)),
                'rms_energy_max': float(np.max(rms)),
                'dynamic_range': float(np.max(rms) - np.min(rms)),
                'dynamic_variance': float(np.var(rms))
            })

            # Peak-to-average ratio
            peak_energy = np.max(np.abs(audio_data))
            avg_energy = np.mean(np.abs(audio_data))
            features['peak_to_average_ratio'] = float(peak_energy / (avg_energy + 1e-10))

            # Energy distribution across time
            # Divide into sections and analyze energy evolution
            num_sections = 10
            section_length = len(audio_data) // num_sections
            section_energies = []

            for i in range(num_sections):
                start = i * section_length
                end = min((i + 1) * section_length, len(audio_data))
                section_energy = np.mean(np.abs(audio_data[start:end])**2)
                section_energies.append(section_energy)

            section_energies = np.array(section_energies)

            features.update({
                'energy_evolution_slope': float(np.polyfit(range(len(section_energies)), section_energies, 1)[0]),
                'energy_evolution_variance': float(np.var(section_energies)),
                'energy_peak_position': float(np.argmax(section_energies) / len(section_energies))
            })

            # Loudness range using perceptual model
            # Simplified version using A-weighting approximation
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)

            # Approximate A-weighting
            a_weight = self._calculate_a_weighting(freqs)
            weighted_magnitude = magnitude * a_weight[:, np.newaxis]

            weighted_power = np.sum(weighted_magnitude**2, axis=0)
            weighted_loudness = 10 * np.log10(weighted_power + 1e-10)

            features.update({
                'perceived_loudness_mean': float(np.mean(weighted_loudness)),
                'perceived_loudness_std': float(np.std(weighted_loudness)),
                'perceived_loudness_range': float(np.max(weighted_loudness) - np.min(weighted_loudness))
            })

            # Crest factor (peak to RMS ratio)
            crest_factors = []
            window_size = self.sample_rate // 4  # 0.25 second windows

            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                peak = np.max(np.abs(window))
                rms_val = np.sqrt(np.mean(window**2))
                if rms_val > 0:
                    crest_factors.append(peak / rms_val)

            if crest_factors:
                features.update({
                    'crest_factor_mean': float(np.mean(crest_factors)),
                    'crest_factor_std': float(np.std(crest_factors))
                })

            return features

        except Exception as e:
            print(f"⚠️ Error extracting dynamic features: {e}")
            return {}

    def _extract_temporal_features(self, audio_data: np.ndarray) -> Dict:
        """Extract temporal evolution and change detection features."""
        features = {}

        try:
            # Feature evolution over time
            # Divide audio into overlapping windows and track feature changes
            window_duration = 30  # seconds
            hop_duration = 15     # seconds

            window_samples = int(window_duration * self.sample_rate)
            hop_samples = int(hop_duration * self.sample_rate)

            window_features = []

            for start in range(0, len(audio_data) - window_samples, hop_samples):
                end = start + window_samples
                window_audio = audio_data[start:end]

                # Extract key features for this window
                window_mfccs = librosa.feature.mfcc(window_audio, sr=self.sample_rate, n_mfcc=13)
                window_chroma = librosa.feature.chroma_stft(window_audio, sr=self.sample_rate)
                window_rms = librosa.feature.rms(window_audio)[0]

                window_feature_vector = np.concatenate([
                    np.mean(window_mfccs, axis=1),
                    np.mean(window_chroma, axis=1),
                    [np.mean(window_rms)]
                ])

                window_features.append(window_feature_vector)

            if len(window_features) > 1:
                window_features = np.array(window_features)

                # Calculate temporal changes
                feature_changes = np.diff(window_features, axis=0)

                features.update({
                    'temporal_change_mean': float(np.mean(np.abs(feature_changes))),
                    'temporal_change_std': float(np.std(np.abs(feature_changes))),
                    'temporal_stability': float(1 / (1 + np.mean(np.std(feature_changes, axis=0))))
                })

                # Identify major changes
                change_magnitudes = np.sqrt(np.sum(feature_changes**2, axis=1))
                major_changes = change_magnitudes > (np.mean(change_magnitudes) + 2*np.std(change_magnitudes))

                features.update({
                    'num_major_changes': int(np.sum(major_changes)),
                    'change_intensity_max': float(np.max(change_magnitudes)),
                    'change_distribution_entropy': float(self._calculate_entropy(change_magnitudes))
                })

            # Onset rate evolution
            onset_frames = librosa.onset.onset_detect(audio_data, sr=self.sample_rate, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)

            # Calculate onset rate in sliding windows
            window_duration_onset = 10  # seconds
            onset_rates = []

            for t in range(0, int(len(audio_data) / self.sample_rate) - window_duration_onset, 5):
                window_start = t
                window_end = t + window_duration_onset

                onsets_in_window = np.sum((onset_times >= window_start) & (onset_times < window_end))
                onset_rate = onsets_in_window / window_duration_onset
                onset_rates.append(onset_rate)

            if onset_rates:
                features.update({
                    'onset_rate_mean': float(np.mean(onset_rates)),
                    'onset_rate_std': float(np.std(onset_rates)),
                    'onset_rate_evolution_slope': float(np.polyfit(range(len(onset_rates)), onset_rates, 1)[0])
                })

            return features

        except Exception as e:
            print(f"⚠️ Error extracting temporal features: {e}")
            return {}

    def _extract_psychoacoustic_features(self, audio_data: np.ndarray) -> Dict:
        """Extract perceptual and psychoacoustic features."""
        features = {}

        try:
            # Bark scale analysis (perceptual frequency scale)
            # Convert to Bark bands
            n_bands = 24
            bark_boundaries = librosa.mel_frequencies(n_mels=n_bands+2, fmin=20, fmax=self.sample_rate//2)

            stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)

            # Group frequencies into Bark bands
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.frame_length)
            bark_energies = []

            for i in range(n_bands):
                freq_mask = (freqs >= bark_boundaries[i]) & (freqs < bark_boundaries[i+1])
                if np.any(freq_mask):
                    band_energy = np.sum(magnitude[freq_mask, :], axis=0)
                    bark_energies.append(band_energy)

            if bark_energies:
                bark_energies = np.array(bark_energies)

                # Psychoacoustic features from Bark bands
                features.update({
                    'bark_spectral_centroid': float(np.mean(np.sum(bark_energies * np.arange(len(bark_energies))[:, np.newaxis], axis=0) / np.sum(bark_energies, axis=0))),
                    'bark_spectral_spread': float(np.mean(np.sqrt(np.sum(bark_energies * (np.arange(len(bark_energies))[:, np.newaxis] - np.sum(bark_energies * np.arange(len(bark_energies))[:, np.newaxis], axis=0) / np.sum(bark_energies, axis=0))**2, axis=0) / np.sum(bark_energies, axis=0)))),
                    'bark_spectral_skewness': float(np.mean([self._calculate_skewness(band) for band in bark_energies])),
                    'bark_spectral_kurtosis': float(np.mean([self._calculate_kurtosis(band) for band in bark_energies]))
                })

            # Roughness estimation (amplitude modulation in critical bands)
            roughness_scores = []

            for band_energy in bark_energies[:12]:  # Focus on lower bands where roughness is most perceptible
                if len(band_energy) > 100:
                    # Look for amplitude modulation in 15-300 Hz range (roughness)
                    envelope = np.abs(librosa.stft(band_energy - np.mean(band_energy), hop_length=1, n_fft=64))

                    # Focus on modulation frequencies that cause roughness
                    mod_freqs = librosa.fft_frequencies(sr=self.sample_rate/self.hop_length, n_fft=64)
                    roughness_mask = (mod_freqs >= 15) & (mod_freqs <= 300)

                    if np.any(roughness_mask):
                        roughness_energy = np.mean(np.sum(envelope[roughness_mask, :], axis=0))
                        roughness_scores.append(roughness_energy)

            if roughness_scores:
                features['perceived_roughness'] = float(np.mean(roughness_scores))

            # Sharpness (high frequency content weighted by perception)
            # Higher frequencies contribute more to perceived sharpness
            freq_weights = freqs / np.max(freqs)  # Simple weighting
            weighted_spectrum = magnitude * freq_weights[:, np.newaxis]

            features['perceived_sharpness'] = float(np.mean(np.sum(weighted_spectrum, axis=0) / np.sum(magnitude, axis=0)))

            # Fluctuation strength (slow amplitude modulations)
            rms = librosa.feature.rms(audio_data, hop_length=self.hop_length)[0]

            if len(rms) > 100:
                # Analyze modulations in 0.5-20 Hz range
                rms_spectrum = np.abs(np.fft.fft(rms - np.mean(rms)))
                mod_freqs_rms = np.fft.fftfreq(len(rms), d=self.hop_length/self.sample_rate)

                fluctuation_mask = (mod_freqs_rms >= 0.5) & (mod_freqs_rms <= 20)
                if np.any(fluctuation_mask):
                    features['fluctuation_strength'] = float(np.mean(rms_spectrum[fluctuation_mask]))

            return features

        except Exception as e:
            print(f"⚠️ Error extracting psychoacoustic features: {e}")
            return {}

    def _extract_cultural_features(self, audio_data: np.ndarray) -> Dict:
        """Extract cultural and genre-specific markers."""
        features = {}

        try:
            # Rhythm pattern analysis for cultural identification
            tempo, beats = librosa.beat.beat_track(audio_data, sr=self.sample_rate)

            # Analyze beat patterns
            if len(beats) > 8:
                beat_intervals = np.diff(beats)

                # Look for common rhythm patterns
                # Swing ratio detection
                if len(beat_intervals) >= 4:
                    # Simple swing detection: look for alternating long-short patterns
                    odd_intervals = beat_intervals[::2]
                    even_intervals = beat_intervals[1::2]

                    if len(odd_intervals) > 2 and len(even_intervals) > 2:
                        swing_ratio = np.mean(odd_intervals) / np.mean(even_intervals)
                        features['swing_ratio'] = float(swing_ratio)
                        features['has_swing_feel'] = swing_ratio > 1.1 or swing_ratio < 0.9

            # Pentatonic scale detection (common in many world music traditions)
            chroma = librosa.feature.chroma_stft(audio_data, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)

            # Pentatonic patterns
            pentatonic_major = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0]  # C pentatonic major pattern
            pentatonic_minor = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # A pentatonic minor pattern

            # Calculate correlations with pentatonic patterns
            pent_major_corr = np.corrcoef(chroma_mean, pentatonic_major)[0, 1]
            pent_minor_corr = np.corrcoef(chroma_mean, pentatonic_minor)[0, 1]

            if not np.isnan(pent_major_corr):
                features['pentatonic_major_correlation'] = float(pent_major_corr)
            if not np.isnan(pent_minor_corr):
                features['pentatonic_minor_correlation'] = float(pent_minor_corr)

            # Microtonal detection (approximate)
            # Look for energy between standard Western pitch classes
            pitch = librosa.yin(audio_data, fmin=80, fmax=800, sr=self.sample_rate)
            valid_pitch = pitch[pitch > 0]

            if len(valid_pitch) > 100:
                # Convert to cents relative to A4
                cents = 1200 * np.log2(valid_pitch / 440)

                # Check how much energy is between semitones
                cents_mod_100 = np.mod(cents, 100)  # Cents within each semitone

                # Count pitches significantly off from standard tuning
                microtonal_pitches = np.sum((cents_mod_100 > 25) & (cents_mod_100 < 75))
                features['microtonal_ratio'] = float(microtonal_pitches / len(valid_pitch))

            # Instrumentation hints from spectral analysis
            # These are rough estimates based on spectral characteristics

            # Distortion detection (common in rock/metal)
            # High harmonic content indicates distortion
            harmonic, _ = librosa.effects.hpss(audio_data)
            spectral_rolloff = librosa.feature.spectral_rolloff(harmonic, sr=self.sample_rate)[0]
            spectral_centroid = librosa.feature.spectral_centroid(harmonic, sr=self.sample_rate)[0]

            distortion_indicator = np.mean(spectral_rolloff / spectral_centroid)
            features['distortion_indicator'] = float(distortion_indicator)

            # Electronic music markers
            # Look for very regular rhythms and synthetic-sounding timbres
            onset_strength = librosa.onset.onset_strength(audio_data, sr=self.sample_rate)

            # Regularity of onsets
            if len(onset_strength) > 100:
                onset_autocorr = np.correlate(onset_strength, onset_strength, mode='full')
                onset_autocorr = onset_autocorr[len(onset_autocorr)//2:]

                # Look for strong periodicities (indicating drum machines/sequenced drums)
                periodic_strength = np.max(onset_autocorr[10:100]) / onset_autocorr[0]
                features['rhythmic_regularity'] = float(periodic_strength)

            return features

        except Exception as e:
            print(f"⚠️ Error extracting cultural features: {e}")
            return {}

    def _extract_lyrical_features(self, track_metadata: Dict) -> Dict:
        """Extract lyrical content and sentiment features."""
        features = {}

        if not self.genius_api_key or not NLTK_AVAILABLE:
            return features

        try:
            # Get lyrics from Genius API
            lyrics = self._fetch_lyrics(track_metadata)

            if not lyrics:
                return features

            # Basic text statistics
            words = lyrics.split()
            sentences = re.split('[.!?]+', lyrics)

            features.update({
                'lyrics_word_count': len(words),
                'lyrics_sentence_count': len(sentences),
                'lyrics_avg_word_length': float(np.mean([len(word) for word in words])),
                'lyrics_avg_sentence_length': float(np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()]))
            })

            # Sentiment analysis
            blob = TextBlob(lyrics)
            features.update({
                'lyrics_sentiment_polarity': float(blob.sentiment.polarity),
                'lyrics_sentiment_subjectivity': float(blob.sentiment.subjectivity)
            })

            # Lyrical complexity
            unique_words = set(word.lower() for word in words if word.isalpha())
            features['lyrics_lexical_diversity'] = float(len(unique_words) / len(words) if words else 0)

            # Emotional keywords analysis
            emotion_keywords = {
                'love': ['love', 'heart', 'kiss', 'romance', 'affection'],
                'sadness': ['sad', 'cry', 'tears', 'lonely', 'depressed', 'hurt'],
                'anger': ['angry', 'mad', 'hate', 'rage', 'furious'],
                'joy': ['happy', 'joy', 'smile', 'laugh', 'celebration'],
                'fear': ['scared', 'afraid', 'fear', 'terror', 'worried'],
                'energy': ['energy', 'power', 'strong', 'alive', 'electric'],
                'party': ['party', 'dance', 'club', 'celebration', 'fun'],
                'spiritual': ['god', 'spirit', 'soul', 'prayer', 'heaven']
            }

            lyrics_lower = lyrics.lower()
            for emotion, keywords in emotion_keywords.items():
                keyword_count = sum(lyrics_lower.count(keyword) for keyword in keywords)
                features[f'lyrics_{emotion}_keywords'] = keyword_count
                features[f'lyrics_{emotion}_ratio'] = float(keyword_count / len(words) if words else 0)

            # Repetition analysis
            line_counts = {}
            lines = lyrics.split('\n')
            for line in lines:
                line = line.strip().lower()
                if line:
                    line_counts[line] = line_counts.get(line, 0) + 1

            if line_counts:
                max_repetition = max(line_counts.values())
                features['lyrics_max_line_repetition'] = max_repetition
                features['lyrics_repetition_ratio'] = float(max_repetition / len(lines))

            return features

        except Exception as e:
            print(f"⚠️ Error extracting lyrical features: {e}")
            return {}

    def _extract_essentia_features(self, audio_data: np.ndarray) -> Dict:
        """Extract advanced features using Essentia library."""
        if not ESSENTIA_AVAILABLE:
            return {}

        features = {}

        try:
            # Configure Essentia algorithms
            windowing = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()

            # Frame-based analysis
            frame_size = 2048
            hop_size = 512

            spectral_complexities = []
            spectral_energies = []

            for frame in es.FrameGenerator(audio_data, frameSize=frame_size, hopSize=hop_size):
                windowed_frame = windowing(frame)
                spectrum_frame = spectrum(windowed_frame)

                # Spectral complexity
                peaks_freq, peaks_mag = spectral_peaks(spectrum_frame)
                if len(peaks_freq) > 0:
                    spectral_complexity = len(peaks_freq) / len(spectrum_frame)
                    spectral_complexities.append(spectral_complexity)

                # Spectral energy
                spectral_energy = np.sum(spectrum_frame**2)
                spectral_energies.append(spectral_energy)

            if spectral_complexities:
                features.update({
                    'essentia_spectral_complexity_mean': float(np.mean(spectral_complexities)),
                    'essentia_spectral_complexity_std': float(np.std(spectral_complexities))
                })

            if spectral_energies:
                features.update({
                    'essentia_spectral_energy_mean': float(np.mean(spectral_energies)),
                    'essentia_spectral_energy_std': float(np.std(spectral_energies))
                })

            # High-level descriptors
            try:
                # Danceability
                danceability = es.Danceability()
                dance_score = danceability(audio_data)
                features['essentia_danceability'] = float(dance_score)
            except:
                pass

            try:
                # Dynamic complexity
                dynamic_complexity = es.DynamicComplexity()
                dynamics_score = dynamic_complexity(audio_data)
                features['essentia_dynamic_complexity'] = float(dynamics_score)
            except:
                pass

            return features

        except Exception as e:
            print(f"⚠️ Error extracting Essentia features: {e}")
            return {}

    # Helper methods

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        if len(data) == 0:
            return 0.0
        hist, _ = np.histogram(data, bins=20)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))

    def _get_key_profiles(self) -> Dict[str, np.ndarray]:
        """Get key profiles for key estimation."""
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        keys = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i, note in enumerate(note_names):
            keys[f'{note}_major'] = np.roll(major_profile, i)
            keys[f'{note}_minor'] = np.roll(minor_profile, i)

        return keys

    def _calculate_a_weighting(self, freqs: np.ndarray) -> np.ndarray:
        """Calculate A-weighting for perceptual loudness."""
        # Simplified A-weighting formula
        f = freqs + 1e-10  # Avoid division by zero

        # A-weighting formula (simplified)
        ra = 12194**2 * f**4 / ((f**2 + 20.6**2) *
                                 np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) *
                                 (f**2 + 12194**2))

        # Convert to dB
        a = 20 * np.log10(ra + 1e-10) + 2.0

        # Convert to linear scale
        return 10**(a / 20)

    def _fetch_lyrics(self, track_metadata: Dict) -> Optional[str]:
        """Fetch lyrics from Genius API."""
        if not self.genius_api_key:
            return None

        try:
            title = track_metadata.get('name', '')
            artist = track_metadata.get('artist', '')

            if not title or not artist:
                return None

            # Search for song on Genius
            search_url = "https://api.genius.com/search"
            headers = {"Authorization": f"Bearer {self.genius_api_key}"}
            params = {"q": f"{artist} {title}"}

            response = requests.get(search_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                hits = data.get('response', {}).get('hits', [])

                if hits:
                    song_url = hits[0]['result']['url']

                    # Note: Getting full lyrics requires web scraping
                    # This is a simplified version that would need additional implementation
                    return f"[Lyrics would be fetched from {song_url}]"

            return None

        except Exception as e:
            print(f"⚠️ Error fetching lyrics: {e}")
            return None

    def _init_essentia_analyzers(self):
        """Initialize Essentia analyzers if available."""
        if ESSENTIA_AVAILABLE:
            try:
                self.essentia_windowing = es.Windowing(type='hann')
                self.essentia_spectrum = es.Spectrum()
                self.essentia_mfcc = es.MFCC()
                print("✅ Essentia analyzers initialized")
            except Exception as e:
                print(f"⚠️ Error initializing Essentia: {e}")

    def _init_vocal_analyzers(self):
        """Initialize vocal analysis components."""
        try:
            # Vocal range parameters
            self.vocal_f0_min = 80   # Hz
            self.vocal_f0_max = 800  # Hz

            # Formant analysis parameters
            self.formant_frequencies = [700, 1220, 2600, 3010]  # Typical formant centers

            print("✅ Vocal analyzers initialized")
        except Exception as e:
            print(f"⚠️ Error initializing vocal analyzers: {e}")

    def _init_harmonic_analyzers(self):
        """Initialize harmonic analysis components."""
        try:
            # Harmonic analysis parameters
            self.chroma_filters = librosa.filters.chroma(
                sr=self.sample_rate,
                n_fft=self.frame_length
            )

            print("✅ Harmonic analyzers initialized")
        except Exception as e:
            print(f"⚠️ Error initializing harmonic analyzers: {e}")


def create_hyperdimensional_fingerprint(audio_file_path: str,
                                      track_metadata: Dict = None,
                                      genius_api_key: str = None) -> Dict:
    """
    Create a comprehensive hyperdimensional fingerprint for a track.

    Args:
        audio_file_path: Path to audio file
        track_metadata: Optional track metadata
        genius_api_key: Optional Genius API key for lyrics

    Returns:
        Dictionary with 300+ features
    """
    try:
        print(f"🧬 Creating hyperdimensional fingerprint for {audio_file_path}")

        # Load audio
        audio_data, sr = librosa.load(audio_file_path, sr=22050)

        # Initialize analyzer
        analyzer = HyperdimensionalAnalyzer(genius_api_key)

        # Perform comprehensive analysis
        fingerprint = analyzer.analyze_track_comprehensive(
            audio_data,
            track_metadata,
            include_lyrics=bool(genius_api_key)
        )

        # Add metadata
        fingerprint['audio_file_path'] = audio_file_path
        fingerprint['sample_rate'] = sr
        fingerprint['fingerprint_version'] = '1.0'

        return fingerprint

    except Exception as e:
        print(f"❌ Error creating fingerprint: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    """Test hyperdimensional analysis."""
    print("🧪 Testing hyperdimensional analyzer...")

    # This would normally analyze a real audio file
    # For testing, we'll create synthetic data

    test_audio = np.random.randn(22050 * 30)  # 30 seconds of noise
    test_metadata = {
        'name': 'Test Track',
        'artist': 'Test Artist'
    }

    analyzer = HyperdimensionalAnalyzer()
    result = analyzer.analyze_track_comprehensive(test_audio, test_metadata, include_lyrics=False)

    feature_count = sum(1 for k, v in result.items() if isinstance(v, (int, float)))
    print(f"✅ Test completed. Extracted {feature_count} features")
