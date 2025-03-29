"""
Melody module for the Infinite Spa Music Generator.

Contains the MelodyGenerator class for creating procedural melodies
that complement chord progressions using voice-leading principles.
"""

import random
from infinitespa.harmony import ChordType

class MelodyGenerator:
    """
    Generates melodic lines that complement a chord progression.
    Uses principles of voice-leading and chord tones to create coherent melodies.
    """
    def __init__(self, key=0, scale_type="major", seed=None):
        """
        Initialize the melody generator.
        
        Args:
            key: Integer representing the key (0=C, 1=C#, etc.)
            scale_type: Type of scale to use ("major" or "minor")
            seed: Random seed for reproducibility
        """
        self.key = key
        self.scale_type = scale_type
        
        # Define scale patterns (intervals from root)
        self.scale_patterns = {
            "major": [0, 2, 4, 5, 7, 9, 11],  # Major scale
            "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor scale
            "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],  # Harmonic minor
            "melodic_minor": [0, 2, 3, 5, 7, 9, 11],   # Melodic minor (ascending)
            "pentatonic_major": [0, 2, 4, 7, 9],       # Major pentatonic
            "pentatonic_minor": [0, 3, 5, 7, 10],      # Minor pentatonic
            "blues": [0, 3, 5, 6, 7, 10]               # Blues scale
        }
        
        # Set the scale based on key and scale type
        self.current_scale = self._build_scale(self.key, self.scale_type)
        
        # Set the melodic range (MIDI note numbers)
        self.lower_bound = 60  # Middle C
        self.upper_bound = 84  # C6
        
        # Initialize the current melodic position
        self.current_position = 72  # C5
        
        # Rhythmic patterns for spa music (in quarter notes)
        # Note: Negative values represent rests
        # 1.0 = quarter note, 2.0 = half note, 3.0 = dotted half note
        # All patterns must sum to 3.0 or less (for 3/4 time)
        self.rhythmic_patterns = [
            # Patterns ending with longer notes to ring out
            [1.0, 2.0],                      # Quarter, half (rings out)
            [-1.0, 2.0],                     # Rest, half note (rings out)
            [1.0, -0.5, 1.5],                # Quarter, short rest, longer note (rings out)
            
            # Single longer notes
            [3.0],                           # Dotted half (whole measure, rings out)
            [2.0, 1.0],                      # Half, quarter
            
            # Sparse patterns with quarter notes and rests
            [1.0, -1.0, 1.0],                # Quarter, rest, quarter
            [-1.0, 1.0, -1.0],               # Rest, quarter, rest
            [2.0, -1.0],                     # Half, rest
            
            # More sparse patterns
            [-1.0, 1.0, 1.0],                # Rest, two quarters
            [-2.0, 1.0],                     # Half rest, quarter
            
            # Very sparse patterns (mostly rests)
            [-2.0, -1.0],                    # Silence for the whole measure
            [-1.0, -1.0, 1.0],               # Two rests, quarter
        ]
        
        # Initialize the random seed
        if seed is not None:
            random.seed(seed)
    
    def _build_scale(self, key, scale_type):
        """Build a scale based on key and scale type."""
        if scale_type not in self.scale_patterns:
            scale_type = "major"  # Default to major if invalid
            
        intervals = self.scale_patterns[scale_type]
        return [(key + interval) % 12 for interval in intervals]
    
    def _is_chord_tone(self, note, chord_notes):
        """Check if a note is a chord tone."""
        return (note % 12) in chord_notes
    
    def _is_dissonant_with_chord(self, note, chord_voicing):
        """
        Check if a note creates dissonant intervals with any chord tone.
        Dissonant intervals are minor 2nds (1 semitone), major 7ths (11 semitones),
        and tritones (6 semitones).
        """
        note_pc = note % 12
        dissonant_intervals = [1, 6, 11]
        
        for chord_note in chord_voicing:
            chord_note_pc = chord_note % 12
            interval = min((note_pc - chord_note_pc) % 12, (chord_note_pc - note_pc) % 12)
            if interval in dissonant_intervals:
                return True
                
        return False
    
    def _get_chord_scale(self, chord):
        """Get appropriate scale for a given chord."""
        root = chord["root"]
        chord_type = chord["type"]
        
        # Choose scale based on chord type
        if chord_type in [ChordType.MAJOR, ChordType.MAJOR7]:
            return self._build_scale(root, "major")
        elif chord_type in [ChordType.MINOR, ChordType.MINOR7]:
            return self._build_scale(root, "minor")
        elif chord_type == ChordType.DOMINANT7:
            # For dominant chords, mixolydian mode (major with flat 7)
            # We can use the major scale of the 4th degree
            return self._build_scale((root + 5) % 12, "major")
        elif chord_type in [ChordType.DIMINISHED, ChordType.DIMINISHED7, ChordType.HALF_DIMINISHED]:
            # For diminished chords, use harmonic minor from the 3rd below
            return self._build_scale((root - 3) % 12, "harmonic_minor")
        elif chord_type == ChordType.AUGMENTED:
            # For augmented chords, use whole tone scale (approximated with major)
            return self._build_scale(root, "major")
        else:
            # Default to major
            return self._build_scale(root, "major")
    
    def _get_next_note_options(self, chord, prev_note, scale):
        """Get possible next notes based on current chord and previous note."""
        options = []
        weights = []
        
        # Get chord tones
        chord_tones = chord["notes"]
        
        # Consider notes in the appropriate range
        for midi_note in range(self.lower_bound, self.upper_bound + 1):
            note_pc = midi_note % 12
            
            # Skip notes that are too far from previous note (avoid large leaps)
            if prev_note is not None and abs(midi_note - prev_note) > 12:
                continue
                
            # Skip notes that create strong dissonances with the chord
            if self._is_dissonant_with_chord(midi_note, chord["voicing"]):
                continue
                
            # Calculate weight based on various factors
            weight = 1.0
            
            # Favor notes in the scale
            if note_pc in scale:
                weight *= 3.0
            else:
                weight *= 0.2  # Non-scale tones are less likely
                
            # Favor chord tones
            if self._is_chord_tone(midi_note, chord_tones):
                weight *= 4.0
                
            # Favor stepwise motion (smaller intervals from previous note)
            if prev_note is not None:
                interval = abs(midi_note - prev_note)
                if interval == 0:  # Same note
                    weight *= 0.5  # Slightly discourage repeating the same note
                elif interval <= 2:  # Step (1-2 semitones)
                    weight *= 2.0
                elif interval <= 4:  # Third (3-4 semitones)
                    weight *= 1.5
                elif interval <= 7:  # Fifth (5-7 semitones)
                    weight *= 1.0
                else:  # Larger intervals
                    weight *= 0.5
                    
            # Favor notes in the middle of the range
            distance_from_center = abs(midi_note - (self.lower_bound + self.upper_bound) // 2)
            range_size = (self.upper_bound - self.lower_bound) // 2
            normalized_distance = distance_from_center / range_size
            weight *= (1.0 - 0.5 * normalized_distance)  # Reduce weight for extreme ranges
            
            options.append(midi_note)
            weights.append(weight)
            
        # If no options, return a default note in the middle of the range
        if not options:
            return [(self.lower_bound + self.upper_bound) // 2], [1.0]
            
        return options, weights
    
    def _choose_next_note(self, options):
        """Choose the next note based on weighted options."""
        notes, weights = options
        
        if not notes:
            return None
            
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Choose next note
        return random.choices(notes, normalized_weights)[0]
    
    def _get_rhythmic_pattern(self, measure_duration):
        """Get a rhythmic pattern that fits within the measure."""
        # Filter patterns that fit within the measure duration
        valid_patterns = []
        for pattern in self.rhythmic_patterns:
            if sum(abs(x) for x in pattern) <= measure_duration:
                valid_patterns.append(pattern)
                
        # If no valid patterns, use a simple quarter note
        if not valid_patterns:
            return [1.0]
            
        # Choose a pattern randomly
        return random.choice(valid_patterns)
    
    def generate_melody_for_chord(self, chord, measure_duration):
        """
        Generate a melodic line for a single chord.
        
        Args:
            chord: Chord dictionary with root, type, notes, and voicing
            measure_duration: Duration of the measure in quarter notes
            
        Returns:
            List of (note, start_time, duration) tuples
        """
        # Get appropriate scale for this chord
        scale = self._get_chord_scale(chord)
        
        # Get rhythmic pattern for this measure
        rhythm = self._get_rhythmic_pattern(measure_duration)
        
        # Generate melody
        melody = []
        current_time = 0
        prev_note = self.current_position  # Start from current position
        
        for duration in rhythm:
            if duration > 0:  # Note
                # Get options for next note
                options = self._get_next_note_options(chord, prev_note, scale)
                
                # Choose next note
                next_note = self._choose_next_note(options)
                
                # Add note to melody
                melody.append((next_note, current_time, duration))
                
                # Update current position
                self.current_position = next_note
                prev_note = next_note
            else:  # Rest
                # Add rest to melody (represented by None)
                melody.append((None, current_time, abs(duration)))
                
            # Update time
            current_time += abs(duration)
            
        return melody
