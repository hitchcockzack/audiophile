"""
Magic Recommendation Narrator

Provides intelligent, contextual explanations for music recommendations.
Creates compelling narratives that explain WHY each track was chosen,
adding mystery, purpose, and musical insight to every suggestion.

Instead of saying "Here's a new song," this narrator explains:
- The sonic DNA that connects it to your taste
- The emotional journey it will take you on
- The specific qualities that make it perfect for you
- Cultural and musical context
- Discovery potential and hidden qualities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import random
from datetime import datetime
import json


class MagicRecommendationNarrator:
    """Creates compelling narratives for music recommendations."""

    def __init__(self):
        """Initialize the magic narrator with storytelling templates."""
        self.emotional_descriptors = {
            'high_valence': ['euphoric', 'luminous', 'transcendent', 'radiant', 'uplifting'],
            'low_valence': ['haunting', 'introspective', 'melancholic', 'profound', 'contemplative'],
            'high_energy': ['electrifying', 'pulsating', 'explosive', 'kinetic', 'dynamic'],
            'low_energy': ['ethereal', 'delicate', 'intimate', 'whispered', 'gentle'],
            'high_danceability': ['groove-laden', 'irresistible', 'rhythmically hypnotic', 'body-moving'],
            'acoustic': ['organic', 'raw', 'authentic', 'stripped-down', 'intimate'],
            'electronic': ['synthesized', 'futuristic', 'digitally crafted', 'sonic architecture']
        }

        self.vocal_descriptors = {
            'bright': ['crystalline', 'soaring', 'luminous', 'piercing', 'brilliant'],
            'warm': ['velvety', 'honey-toned', 'embracing', 'comforting', 'golden'],
            'rough': ['gravelly', 'weathered', 'textured', 'raw', 'edgy'],
            'smooth': ['silken', 'polished', 'flowing', 'effortless', 'butter-smooth']
        }

        self.harmonic_descriptors = {
            'complex': ['intricate harmonic tapestry', 'sophisticated chord progressions', 'labyrinthine musical architecture'],
            'simple': ['pure harmonic clarity', 'elegant simplicity', 'crystalline structure'],
            'jazzy': ['sophisticated jazz harmonies', 'complex chord substitutions', 'improvisational spirit'],
            'folk': ['timeless harmonic traditions', 'earthbound progressions', 'ancestral musical wisdom']
        }

        self.discovery_templates = {
            'hidden_gem': [
                "💎 You're entering uncharted territory.",
                "🔍 This is a rare find that's been waiting for you.",
                "✨ Few have discovered this sonic treasure.",
                "🎯 You've unlocked a musical secret."
            ],
            'rising_star': [
                "🌟 This track is climbing the charts in its own universe.",
                "🚀 You're catching this one on its ascent to greatness.",
                "📈 This song is building momentum in its artist's catalog.",
                "⭐ You're witnessing a star being born."
            ],
            'perfect_match': [
                "🎯 This track was practically made for your ears.",
                "🧬 The sonic DNA is a perfect match for your taste.",
                "🔮 Our algorithms are 95% certain you'll love this.",
                "✨ This recommendation feels like destiny."
            ],
            'mood_match': [
                "🎭 This captures exactly what you're feeling right now.",
                "🌊 This track flows perfectly with your current vibe.",
                "🎨 The emotional palette matches your mood perfectly.",
                "⚡ This energy is precisely what you need."
            ]
        }

        self.connection_templates = [
            "This track shares the {similarity_type} of '{reference_track}', with the {unique_quality} of '{comparison_track}'",
            "Drawing from the {musical_element} you love in '{reference_track}', but with a {twist_description}",
            "If '{reference_track}' and '{comparison_track}' had a musical conversation, this would be their beautiful conclusion",
            "This captures the essence of what makes '{reference_track}' special, elevated with {enhancement_quality}"
        ]

        self.revelation_templates = [
            "Listen for the way the {audio_feature} {behavior} at {timestamp} - it's pure magic.",
            "Pay attention to how the {element} builds tension before the {resolution} - masterful composition.",
            "The {unique_element} will surprise you around {timestamp} - unexpected but perfect.",
            "Notice the subtle {detail} - it's what separates this from the ordinary."
        ]

    def narrate_recommendation(self, track_data: Dict, context: Dict) -> Dict:
        """
        Create a magical narrative for a track recommendation.

        Args:
            track_data: Track information with audio features and metadata
            context: Context about why this was recommended (similarity, mood, etc.)

        Returns:
            Dictionary with narrative components
        """
        try:
            # Extract key information
            track_name = track_data.get('name', 'Unknown Track')
            artist_name = track_data.get('artist', 'Unknown Artist')

            # Analyze audio characteristics
            audio_profile = self._analyze_audio_characteristics(track_data)

            # Determine recommendation type and context
            rec_type = self._determine_recommendation_type(track_data, context)

            # Generate narrative components
            narrative = {
                'opening': self._create_opening(rec_type, audio_profile),
                'sonic_description': self._describe_sonic_characteristics(audio_profile, track_data),
                'emotional_journey': self._describe_emotional_journey(audio_profile, track_data),
                'discovery_insight': self._create_discovery_insight(track_data, context),
                'connection_story': self._create_connection_story(track_data, context),
                'listening_instruction': self._create_listening_instruction(audio_profile),
                'full_narrative': '',
                'metadata': {
                    'track_name': track_name,
                    'artist_name': artist_name,
                    'recommendation_type': rec_type,
                    'confidence': context.get('confidence', 0.8),
                    'generated_at': datetime.now().isoformat()
                }
            }

            # Combine into full narrative
            narrative['full_narrative'] = self._combine_narrative(narrative)

            return narrative

        except Exception as e:
            # Fallback to simple description
            return {
                'full_narrative': f"Discover '{track_name}' by {artist_name} - a track carefully selected for your unique taste.",
                'error': str(e)
            }

    def _analyze_audio_characteristics(self, track_data: Dict) -> Dict:
        """Analyze audio characteristics to create descriptive profile."""
        try:
            # Extract audio features
            energy = track_data.get('energy', 0.5)
            valence = track_data.get('valence', 0.5)
            danceability = track_data.get('danceability', 0.5)
            acousticness = track_data.get('acousticness', 0.5)
            instrumentalness = track_data.get('instrumentalness', 0.5)
            speechiness = track_data.get('speechiness', 0.1)
            tempo = track_data.get('tempo', 120)

            # Enhanced features if available
            spectral_centroid = track_data.get('spectral_centroid_mean', 0.5)
            vocal_brightness = track_data.get('vocal_brightness', 0.5)
            harmonic_complexity = track_data.get('harmonic_complexity', 0.5)

            profile = {
                'energy_level': self._categorize_continuous(energy, ['gentle', 'moderate', 'intense']),
                'emotional_tone': self._categorize_continuous(valence, ['melancholic', 'contemplative', 'euphoric']),
                'rhythmic_pull': self._categorize_continuous(danceability, ['meditative', 'engaging', 'irresistible']),
                'sonic_texture': 'acoustic' if acousticness > 0.6 else 'electronic' if acousticness < 0.2 else 'hybrid',
                'vocal_presence': 'instrumental' if instrumentalness > 0.7 else 'vocal-driven' if speechiness > 0.3 else 'melodic',
                'tempo_feel': self._categorize_tempo(tempo),
                'brightness': self._categorize_continuous(spectral_centroid, ['warm', 'balanced', 'brilliant']),
                'vocal_quality': self._determine_vocal_quality(track_data),
                'harmonic_sophistication': self._categorize_continuous(harmonic_complexity, ['pure', 'intricate', 'complex'])
            }

            return profile

        except Exception as e:
            return {'error': str(e)}

    def _determine_recommendation_type(self, track_data: Dict, context: Dict) -> str:
        """Determine the type of recommendation for narrative selection."""

        # Check for discovery indicators
        discovery_score = track_data.get('discovery_score', 0)
        momentum_score = track_data.get('momentum_score', 0)
        similarity_score = track_data.get('similarity_score', 0)
        is_hidden_gem = track_data.get('is_hidden_gem', False)
        is_rising_star = track_data.get('is_rising_star', False)

        if is_hidden_gem:
            return 'hidden_gem'
        elif is_rising_star:
            return 'rising_star'
        elif similarity_score > 0.9:
            return 'perfect_match'
        elif context.get('mood'):
            return 'mood_match'
        elif discovery_score > 0.7:
            return 'discovery'
        else:
            return 'personalized'

    def _create_opening(self, rec_type: str, audio_profile: Dict) -> str:
        """Create an engaging opening line."""

        if rec_type in self.discovery_templates:
            opening = random.choice(self.discovery_templates[rec_type])
        else:
            # Create dynamic opening based on audio characteristics
            energy_desc = audio_profile.get('energy_level', 'moderate')
            emotional_desc = audio_profile.get('emotional_tone', 'contemplative')

            openings = [
                f"🎵 Prepare for a {energy_desc}, {emotional_desc} journey.",
                f"✨ This {emotional_desc} masterpiece is calling your name.",
                f"🌟 Your musical DNA has led us to this {energy_desc} revelation.",
                f"🎯 The perfect convergence of {energy_desc} energy and {emotional_desc} soul."
            ]
            opening = random.choice(openings)

        return opening

    def _describe_sonic_characteristics(self, audio_profile: Dict, track_data: Dict) -> str:
        """Describe the sonic characteristics in evocative language."""

        descriptions = []

        # Energy and emotion
        energy = audio_profile.get('energy_level', 'moderate')
        emotion = audio_profile.get('emotional_tone', 'contemplative')
        texture = audio_profile.get('sonic_texture', 'hybrid')

        if energy == 'intense' and emotion == 'euphoric':
            descriptions.append("This track radiates pure kinetic energy, wrapped in luminous euphoria")
        elif energy == 'gentle' and emotion == 'melancholic':
            descriptions.append("Delicate sonic threads weave a tapestry of beautiful melancholy")
        elif texture == 'acoustic':
            descriptions.append("Raw, organic instrumentation creates an intimate sonic sanctuary")
        elif texture == 'electronic':
            descriptions.append("Synthesized soundscapes build a futuristic audio architecture")

        # Vocal characteristics
        vocal_quality = audio_profile.get('vocal_quality', 'balanced')
        vocal_presence = audio_profile.get('vocal_presence', 'melodic')

        if vocal_presence == 'vocal-driven' and vocal_quality:
            vocal_desc = random.choice(self.vocal_descriptors.get(vocal_quality, ['expressive']))
            descriptions.append(f"The {vocal_desc} vocals carry profound emotional weight")
        elif vocal_presence == 'instrumental':
            descriptions.append("Pure instrumental expression speaks without words")

        # Rhythmic elements
        rhythmic_pull = audio_profile.get('rhythmic_pull', 'engaging')
        tempo_feel = audio_profile.get('tempo_feel', 'moderate')

        if rhythmic_pull == 'irresistible':
            descriptions.append(f"The {tempo_feel} groove creates an irresistible rhythmic magnetism")

        return '. '.join(descriptions) + '.'

    def _describe_emotional_journey(self, audio_profile: Dict, track_data: Dict) -> str:
        """Describe the emotional journey the track will provide."""

        emotional_tone = audio_profile.get('emotional_tone', 'contemplative')
        energy_level = audio_profile.get('energy_level', 'moderate')
        harmonic_sophistication = audio_profile.get('harmonic_sophistication', 'intricate')

        journey_templates = {
            'euphoric_intense': "This track will lift your spirit and energize your soul, creating moments of pure transcendence.",
            'melancholic_gentle': "Prepare for a journey into introspective beauty, where vulnerability becomes strength.",
            'contemplative_moderate': "This composition invites deep reflection while maintaining perfect emotional balance.",
            'euphoric_gentle': "Experience joy in its most refined form - uplifting without overwhelming.",
            'melancholic_intense': "Powerful emotions flow through dynamic musical expression, cathartic and transformative."
        }

        journey_key = f"{emotional_tone}_{energy_level}"

        if journey_key in journey_templates:
            journey = journey_templates[journey_key]
        else:
            # Default journey based on primary emotional tone
            if emotional_tone == 'euphoric':
                journey = "This track channels pure positivity, designed to elevate your emotional state."
            elif emotional_tone == 'melancholic':
                journey = "Beautiful sadness transforms into profound understanding through musical expression."
            else:
                journey = "This composition creates space for whatever emotions you bring to it."

        # Add harmonic sophistication context
        if harmonic_sophistication == 'complex':
            journey += " Complex harmonies reveal new layers with each listen."
        elif harmonic_sophistication == 'pure':
            journey += " Pure, crystalline harmonies create immediate emotional connection."

        return journey

    def _create_discovery_insight(self, track_data: Dict, context: Dict) -> str:
        """Create insight about why this is a great discovery."""

        # Check for discovery indicators
        discovery_score = track_data.get('discovery_score', 0)
        momentum_score = track_data.get('momentum_score', 0)
        relative_popularity = track_data.get('relative_popularity', {})

        insights = []

        if discovery_score > 0.7:
            insights.append(f"This track scores {discovery_score:.0%} on our discovery scale - truly special.")

        if momentum_score > 0.7:
            insights.append(f"With {momentum_score:.0%} momentum, this song is gaining recognition it deserves.")

        if relative_popularity:
            artist_rank = relative_popularity.get('popularity_rank', 0)
            catalog_size = relative_popularity.get('artist_catalog_size', 0)

            if artist_rank and catalog_size:
                if artist_rank <= 3:
                    insights.append(f"This sits in the top 3 of the artist's {catalog_size}-track catalog.")
                elif artist_rank <= catalog_size * 0.2:
                    insights.append(f"This hidden gem ranks #{artist_rank} in the artist's extensive catalog.")

        # Check for uniqueness
        if track_data.get('is_hidden_gem'):
            insights.append("Most people haven't discovered this yet - you're among the first.")

        if track_data.get('is_rising_star'):
            insights.append("This track is climbing its way to becoming a classic.")

        if not insights:
            confidence = context.get('confidence', 0.8)
            insights.append(f"Our algorithms are {confidence:.0%} confident this matches your taste perfectly.")

        return ' '.join(insights)

    def _create_connection_story(self, track_data: Dict, context: Dict) -> str:
        """Create a story about how this connects to user's existing taste."""

        reference_tracks = context.get('reference_tracks', [])
        similarity_score = track_data.get('similarity_score', 0)

        if reference_tracks and len(reference_tracks) > 0:
            reference_track = reference_tracks[0]

            # Determine what connects them
            connection_elements = []

            # Audio feature connections
            if track_data.get('energy', 0) > 0.7 and similarity_score > 0.8:
                connection_elements.append("driving energy")

            if track_data.get('valence', 0) > 0.7:
                connection_elements.append("uplifting spirit")
            elif track_data.get('valence', 0) < 0.3:
                connection_elements.append("introspective depth")

            if track_data.get('acousticness', 0) > 0.6:
                connection_elements.append("organic instrumentation")
            elif track_data.get('acousticness', 0) < 0.2:
                connection_elements.append("electronic sophistication")

            # Create connection narrative
            if connection_elements:
                main_connection = connection_elements[0]
                story = f"This track shares the {main_connection} you love, while exploring new sonic territory."
            else:
                story = f"While different in surface details, this captures the essence of what draws you to great music."
        else:
            # General connection story
            story = "This recommendation emerges from deep analysis of your musical preferences and listening patterns."

        return story

    def _create_listening_instruction(self, audio_profile: Dict) -> str:
        """Create specific listening instructions or highlights."""

        instructions = []

        # Based on audio characteristics
        vocal_presence = audio_profile.get('vocal_presence', 'melodic')
        energy_level = audio_profile.get('energy_level', 'moderate')
        harmonic_sophistication = audio_profile.get('harmonic_sophistication', 'intricate')

        if vocal_presence == 'vocal-driven':
            instructions.append("Listen closely to the vocal performance - every inflection tells a story.")

        if energy_level == 'intense':
            instructions.append("Let the intensity wash over you - this track rewards full attention.")
        elif energy_level == 'gentle':
            instructions.append("Find a quiet moment for this one - its beauty unfolds in stillness.")

        if harmonic_sophistication == 'complex':
            instructions.append("Multiple listens will reveal hidden harmonic layers.")

        # Add mystique
        mystique_additions = [
            "Trust the journey this track wants to take you on.",
            "Pay attention to your emotional response - it's carefully crafted.",
            "This is more than background music - it's a sonic experience.",
            "Let your analytical mind rest and feel the music."
        ]

        instructions.append(random.choice(mystique_additions))

        return ' '.join(instructions)

    def _combine_narrative(self, narrative_parts: Dict) -> str:
        """Combine all narrative parts into a cohesive story."""

        parts = []

        # Opening
        if narrative_parts.get('opening'):
            parts.append(narrative_parts['opening'])

        # Sonic description
        if narrative_parts.get('sonic_description'):
            parts.append(narrative_parts['sonic_description'])

        # Connection story
        if narrative_parts.get('connection_story'):
            parts.append(narrative_parts['connection_story'])

        # Emotional journey
        if narrative_parts.get('emotional_journey'):
            parts.append(narrative_parts['emotional_journey'])

        # Discovery insight
        if narrative_parts.get('discovery_insight'):
            parts.append(narrative_parts['discovery_insight'])

        # Listening instruction
        if narrative_parts.get('listening_instruction'):
            parts.append(narrative_parts['listening_instruction'])

        # Add final call to action
        track_name = narrative_parts.get('metadata', {}).get('track_name', 'this track')
        parts.append(f"\n🎧 Play {track_name}. Now.")

        return '\n\n'.join(parts)

    # Helper methods

    def _categorize_continuous(self, value: float, categories: List[str]) -> str:
        """Categorize a continuous value into discrete categories."""
        if value <= 0.33:
            return categories[0]
        elif value <= 0.66:
            return categories[1]
        else:
            return categories[2]

    def _categorize_tempo(self, tempo: float) -> str:
        """Categorize tempo into feel categories."""
        if tempo < 70:
            return 'glacial'
        elif tempo < 90:
            return 'relaxed'
        elif tempo < 110:
            return 'moderate'
        elif tempo < 130:
            return 'energetic'
        elif tempo < 150:
            return 'driving'
        else:
            return 'frenetic'

    def _determine_vocal_quality(self, track_data: Dict) -> str:
        """Determine vocal quality from available features."""

        # Use enhanced features if available
        vocal_brightness = track_data.get('vocal_brightness', 0.5)
        vocal_warmth = track_data.get('vocal_warmth', 0.5)
        vocal_roughness = track_data.get('vocal_roughness', 0.5)

        if vocal_brightness > 0.7:
            return 'bright'
        elif vocal_warmth > 0.7:
            return 'warm'
        elif vocal_roughness > 0.7:
            return 'rough'
        else:
            return 'smooth'

    def create_batch_narratives(self, recommendations: List[Dict], context: Dict) -> List[Dict]:
        """Create narratives for a batch of recommendations."""

        narratives = []

        for i, track in enumerate(recommendations):
            # Add index-specific context
            track_context = context.copy()
            track_context['position'] = i + 1
            track_context['total_recommendations'] = len(recommendations)

            # Create narrative
            narrative = self.narrate_recommendation(track, track_context)

            # Add to track data
            enhanced_track = track.copy()
            enhanced_track['narrative'] = narrative

            narratives.append(enhanced_track)

        return narratives

    def create_playlist_narrative(self, playlist_data: Dict, tracks: List[Dict]) -> str:
        """Create a narrative for an entire playlist."""

        playlist_name = playlist_data.get('name', 'Your AI Playlist')
        track_count = len(tracks)

        # Analyze overall playlist characteristics
        if tracks:
            avg_energy = np.mean([t.get('energy', 0.5) for t in tracks])
            avg_valence = np.mean([t.get('valence', 0.5) for t in tracks])
            diversity_score = playlist_data.get('diversity_factor', 0.3)

            # Create playlist story
            energy_desc = self._categorize_continuous(avg_energy, ['contemplative', 'balanced', 'energetic'])
            emotion_desc = self._categorize_continuous(avg_valence, ['introspective', 'reflective', 'uplifting'])

            narrative = f"""
🎵 **{playlist_name}**

This {energy_desc} collection of {track_count} tracks creates an {emotion_desc} musical journey.

Each song has been carefully selected not just for how well it matches your taste, but for how it contributes to the overall flow and emotional arc of the playlist.

With {diversity_score:.0%} diversity, this playlist balances the familiar comfort of your musical preferences with carefully chosen discoveries that will expand your horizons.

The algorithmic curation ensures smooth transitions while maintaining your interest throughout the entire listening experience.

🎧 Press play and let the journey begin.
"""
        else:
            narrative = f"**{playlist_name}** - A carefully curated musical experience awaits."

        return narrative.strip()


def enhance_recommendations_with_narratives(recommendations: List[Dict],
                                          context: Dict = None) -> List[Dict]:
    """
    Enhance a list of recommendations with magical narratives.

    Args:
        recommendations: List of track recommendation dictionaries
        context: Optional context about the recommendation session

    Returns:
        Enhanced recommendations with narrative explanations
    """
    narrator = MagicRecommendationNarrator()

    if context is None:
        context = {
            'session_type': 'general',
            'timestamp': datetime.now().isoformat()
        }

    return narrator.create_batch_narratives(recommendations, context)


if __name__ == "__main__":
    """Test the magic narrator."""
    print("🎭 Testing Magic Recommendation Narrator...")

    # Test track data
    test_track = {
        'name': 'Test Track',
        'artist': 'Test Artist',
        'energy': 0.8,
        'valence': 0.7,
        'danceability': 0.6,
        'acousticness': 0.2,
        'discovery_score': 0.8,
        'is_rising_star': True,
        'similarity_score': 0.9
    }

    test_context = {
        'mood': 'energetic',
        'reference_tracks': ['Previous Track'],
        'confidence': 0.95
    }

    narrator = MagicRecommendationNarrator()
    narrative = narrator.narrate_recommendation(test_track, test_context)

    print("\n" + "="*60)
    print("MAGIC NARRATIVE EXAMPLE:")
    print("="*60)
    print(narrative['full_narrative'])
    print("="*60)

    print("\n✅ Magic Narrator test completed!")
