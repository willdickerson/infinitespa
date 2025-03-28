import random
import numpy as np  # Used for mathematical operations
from enum import Enum
import os
import subprocess
from midiutil import MIDIFile

class ChordType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    DOMINANT7 = "dominant7"
    MAJOR7 = "major7"
    MINOR7 = "minor7"
    HALF_DIMINISHED = "half_diminished"
    DIMINISHED7 = "diminished7"

# Note representation: C=0, C#=1, ..., B=11
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord structures (intervals from root)
CHORD_STRUCTURES = {
    ChordType.MAJOR: [0, 4, 7],            # root, major third, perfect fifth
    ChordType.MINOR: [0, 3, 7],            # root, minor third, perfect fifth  
    ChordType.DIMINISHED: [0, 3, 6],       # root, minor third, diminished fifth
    ChordType.AUGMENTED: [0, 4, 8],        # root, major third, augmented fifth
    ChordType.DOMINANT7: [0, 4, 7, 10],    # root, major third, perfect fifth, minor seventh
    ChordType.MAJOR7: [0, 4, 7, 11],       # root, major third, perfect fifth, major seventh
    ChordType.MINOR7: [0, 3, 7, 10],       # root, minor third, perfect fifth, minor seventh
    ChordType.HALF_DIMINISHED: [0, 3, 6, 10], # root, minor third, diminished fifth, minor seventh
    ChordType.DIMINISHED7: [0, 3, 6, 9],   # root, minor third, diminished fifth, diminished seventh
}

# Functional harmonic relationships - based on Tymoczko's theories
FUNCTION_MAP = {
    # Standard functional relationships in Western harmony
    # Tonic functions
    "I": {"root": 0, "type": ChordType.MAJOR},
    "i": {"root": 0, "type": ChordType.MINOR},
    "III": {"root": 4, "type": ChordType.MAJOR},
    "iii": {"root": 4, "type": ChordType.MINOR},
    "VI": {"root": 9, "type": ChordType.MAJOR},
    "vi": {"root": 9, "type": ChordType.MINOR},
    
    # Subdominant functions
    "IV": {"root": 5, "type": ChordType.MAJOR},
    "iv": {"root": 5, "type": ChordType.MINOR},
    "II": {"root": 2, "type": ChordType.MAJOR},
    "ii": {"root": 2, "type": ChordType.MINOR},
    
    # Dominant functions
    "V": {"root": 7, "type": ChordType.MAJOR},
    "V7": {"root": 7, "type": ChordType.DOMINANT7},
    "vii°": {"root": 11, "type": ChordType.DIMINISHED},
    "VII": {"root": 11, "type": ChordType.MAJOR},
}

# Tymoczko-inspired voice-leading graph
VOICE_LEADING_GRAPH = {
    # Tonic functions can move to subdominant or dominant
    "I": ["I", "iii", "vi", "IV", "ii", "V", "V7", "vii°"],
    "i": ["i", "III", "VI", "iv", "ii", "V", "V7", "vii°"],
    "iii": ["iii", "vi", "I", "IV", "ii", "V", "V7"],
    "vi": ["vi", "I", "iii", "IV", "ii", "V", "V7"],
    
    # Subdominant functions typically move to dominant or tonic
    "IV": ["IV", "ii", "V", "V7", "vii°", "I", "vi", "iii"],
    "iv": ["iv", "ii", "V", "V7", "vii°", "i", "VI", "III"],
    "ii": ["ii", "V", "V7", "vii°", "I", "vi", "IV", "iii"],
    
    # Dominant functions typically resolve to tonic
    "V": ["V", "V7", "I", "i", "vi", "iii", "IV", "ii"],
    "V7": ["V7", "I", "i", "vi", "iii", "IV"],
    "vii°": ["vii°", "I", "i", "vi", "iii", "IV"],
    "VII": ["VII", "III", "i", "VI", "iv"],
    
    # Adding missing chord functions with complete relationships
    "III": ["III", "i", "VI", "iv", "V", "VII", "v", "ii°"],
    "VI": ["VI", "i", "III", "iv", "ii", "V", "VII"],
    "II": ["II", "V", "V7", "I", "vi", "IV", "vii°"],
}

# Voice-leading weights based on Tymoczko's principles (smaller = more preferred)
VOICE_LEADING_WEIGHTS = {
    # Stronger tendencies
    ("V", "I"): 1,
    ("V7", "I"): 0.5,  # Even stronger resolution tendency
    ("vii°", "I"): 0.8,
    ("IV", "I"): 2,
    ("ii", "V"): 1.5,
    ("vi", "ii"): 2,
    
    # Cross-relation weights (based on voice-leading smoothness)
    ("I", "vi"): 1.2,
    ("I", "iii"): 1.5,
    ("I", "IV"): 1.3,
    ("I", "V"): 1.3,
    
    # Default weight for other progressions
    "default": 3
}

class TymoczkoChordGenerator:
    def __init__(self, key=0, seed=None):
        """
        Initialize the chord progression generator.
        
        Args:
            key: Integer representing the key (0=C, 1=C#, etc.)
            seed: Random seed for reproducibility
        """
        self.key = key
        self.current_function = "I"  # Start with tonic
        self.current_chord = None
        self.voicing = [60, 64, 67]  # Starting with close middle-C major triad voicing
        self.history = []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Initialize the first chord
        self._generate_next_chord()
    
    def _get_chord_notes(self, root, chord_type):
        """Get the absolute note values for a chord."""
        intervals = CHORD_STRUCTURES[chord_type]
        
        # For seventh chords, omit the fifth (index 2) to keep as three-note chords
        if chord_type in [ChordType.DOMINANT7, ChordType.MAJOR7, ChordType.MINOR7, 
                          ChordType.HALF_DIMINISHED, ChordType.DIMINISHED7]:
            intervals = [intervals[0], intervals[1], intervals[3]]  # Root, third, seventh
            
        return [(root + interval) % 12 for interval in intervals]
    
    def _calculate_voice_leading_distance(self, chord1, chord2):
        """
        Calculate the voice-leading distance between two chords.
        Based on Tymoczko's principle of efficient voice leading.
        """
        # If chords have different numbers of notes, pad the shorter one
        len1, len2 = len(chord1), len(chord2)
        if len1 < len2:
            chord1 = chord1 + chord1[:len2-len1]
        elif len2 < len1:
            chord2 = chord2 + chord2[:len1-len2]
        
        # Find the optimal voice-leading distance by trying different mappings
        best_distance = float('inf')
        
        # Try a reasonable number of permutations for computational feasibility
        for _ in range(min(100, len(chord2)**2)):
            perm = np.random.permutation(chord2)
            
            # Calculate sum of squared semitone distances (L2 norm)
            distances = []
            for note1, note2 in zip(chord1, perm):
                # Find the shortest distance considering octave equivalence
                dist = min((note1 - note2) % 12, (note2 - note1) % 12)
                distances.append(dist ** 2)
                
            total_distance = sum(distances) ** 0.5
            best_distance = min(best_distance, total_distance)
            
        return best_distance
    
    def _find_optimal_voicing(self, target_notes, prev_voicing):
        """
        Find the optimal voicing for the target notes that minimizes voice leading distance.
        Uses Tymoczko's principle of minimal voice leading motion.
        """
        best_voicing = None
        best_distance = float('inf')
        
        # Range constraints for voices (MIDI note numbers)
        ranges = [
            (48, 72),  # Bass: C2-C4
            (55, 79),  # Tenor: G2-G4
            (60, 84),  # Alto: C3-C5
            (67, 91)   # Soprano: G3-G5
        ]
        
        # Generate all possible voicings within reasonable voice ranges
        # Start with the previous voicing's register
        prev_registers = [note // 12 for note in prev_voicing]
        
        # For computational efficiency, limit to +/- 1 octave from previous
        possible_voicings = []
        
        # Handle different chord sizes
        target_size = len(target_notes)
        prev_size = len(prev_voicing)
        
        # If target chord has more notes than previous, add registers for new voices
        if target_size > prev_size:
            prev_registers.extend([5] * (target_size - prev_size))  # Default to middle register
        
        # Generate possible voicings
        for note_idx, note in enumerate(target_notes):
            if note_idx < len(prev_registers):
                register = prev_registers[note_idx]
                # Try the note in previous register and adjacent registers
                possible_registers = [register-1, register, register+1]
                possible_voicings.append([note + r * 12 for r in possible_registers 
                                         if ranges[min(note_idx, 3)][0] <= note + r * 12 <= ranges[min(note_idx, 3)][1]])
            else:
                # For new notes, try middle registers
                possible_voicings.append([note + r * 12 for r in [4, 5] 
                                         if ranges[min(note_idx, 3)][0] <= note + r * 12 <= ranges[min(note_idx, 3)][1]])
        
        # Generate combinations of different registers (limited for computational feasibility)
        max_combinations = 100
        combinations_count = 0
        
        eval_prev_voicing = prev_voicing
        
        def backtrack(index, current_voicing, eval_prev_voicing_local):
            nonlocal best_voicing, best_distance, combinations_count
            
            if combinations_count >= max_combinations:
                return
                
            if index == len(possible_voicings):
                # Evaluate this voicing
                distance = self._calculate_voice_leading_distance(eval_prev_voicing_local, current_voicing)
                
                # Penalize voicings with voice crossings
                sorted_voicing = sorted(current_voicing)
                if sorted_voicing != current_voicing:
                    distance += 5  # Penalty for voice crossing
                
                if distance < best_distance:
                    best_distance = distance
                    best_voicing = current_voicing.copy()
                
                combinations_count += 1
                return
            
            for note in possible_voicings[index]:
                # Apply voice-leading constraints (e.g., avoid large leaps)
                if index < len(eval_prev_voicing) and abs(note - eval_prev_voicing[index]) > 7:
                    continue  # Skip if leap is too large
                
                current_voicing.append(note)
                backtrack(index + 1, current_voicing, eval_prev_voicing_local)
                current_voicing.pop()
        
        # Find optimal voicing by minimal voice-leading distance
        eval_prev_voicing_local = eval_prev_voicing[:len(target_notes)]
        while len(eval_prev_voicing_local) < len(target_notes):
            # Pad with high notes if needed
            eval_prev_voicing_local.append(eval_prev_voicing_local[-1] + 4)
        
        backtrack(0, [], eval_prev_voicing_local)
        
        # If no valid voicing found (rare), create a simple one
        if best_voicing is None:
            best_voicing = []
            for i, note in enumerate(target_notes):
                register = 5 if i > 0 else 4  # Bass in lower register
                best_voicing.append(note + register * 12)
        
        return best_voicing
    
    def _generate_next_chord(self):
        """Generate the next chord based on voice-leading principles."""
        # Choose next chord function based on weighted probabilities
        possible_next_functions = VOICE_LEADING_GRAPH[self.current_function]
        
        # Calculate weights based on voice-leading tendencies
        weights = []
        for next_func in possible_next_functions:
            pair = (self.current_function, next_func)
            weight = VOICE_LEADING_WEIGHTS.get(pair, VOICE_LEADING_WEIGHTS["default"])
            
            # Apply additional weighting based on recent history to avoid repetition
            if len(self.history) > 0:
                # Avoid immediate repetition
                if next_func == self.current_function:
                    weight *= 2  # Penalize repetition
                
                # Avoid going back and forth between two chords
                if len(self.history) >= 2 and next_func == self.history[-2][0]:
                    weight *= 1.5  # Penalize oscillation
            
            # Convert weight to probability (inverse - lower weight = higher probability)
            weights.append(1.0 / weight)
        
        # Normalize weights to probabilities
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        # Choose next function
        next_function = random.choices(possible_next_functions, probabilities)[0]
        
        # Get chord information for the chosen function
        chord_info = FUNCTION_MAP[next_function]
        chord_root = (self.key + chord_info["root"]) % 12
        chord_type = chord_info["type"]
        
        # Get chord notes (pitch classes)
        chord_notes = self._get_chord_notes(chord_root, chord_type)
        
        # Find optimal voicing based on previous chord
        if self.voicing:
            self.voicing = self._find_optimal_voicing(chord_notes, self.voicing)
        else:
            # Initial voicing if none exists
            self.voicing = [note + 60 + (i * 4) for i, note in enumerate(chord_notes)]
        
        # Store the chord information
        self.current_chord = {
            "function": next_function,
            "root": chord_root,
            "type": chord_type,
            "notes": chord_notes,
            "voicing": self.voicing
        }
        
        # Update current function and history
        self.current_function = next_function
        self.history.append((next_function, chord_root, chord_type))
        
        # Limit history length
        if len(self.history) > 10:
            self.history.pop(0)
            
        return self.current_chord
    
    def get_next_chord(self):
        """
        Get the next chord in the progression.
        Returns a dictionary with chord information.
        """
        return self._generate_next_chord()
    
    def get_chord_name(self, chord_to_name=None):
        """Get the name of the current chord or a specified chord."""
        if chord_to_name is None:
            chord_to_name = self.current_chord
            
        if chord_to_name is None:
            return None
            
        root_name = NOTE_NAMES[chord_to_name["root"]]
        type_str = chord_to_name["type"].value
        
        # Format the chord name
        if type_str == "major":
            return f"{root_name}"
        elif type_str == "minor":
            return f"{root_name}m"
        elif type_str == "diminished":
            return f"{root_name}°"
        elif type_str == "augmented":
            return f"{root_name}+"
        elif type_str == "dominant7":
            return f"{root_name}7"
        elif type_str == "major7":
            return f"{root_name}maj7"
        elif type_str == "minor7":
            return f"{root_name}m7"
        elif type_str == "half_diminished":
            return f"{root_name}ø7"
        elif type_str == "diminished7":
            return f"{root_name}°7"
        else:
            return f"{root_name} {type_str}"
    
    def get_current_state(self):
        """Get the current state of the generator."""
        return {
            "key": NOTE_NAMES[self.key],
            "current_function": self.current_function,
            "current_chord": self.get_chord_name(),
            "voicing": self.voicing,
            "history": [(func, NOTE_NAMES[root], type.value) for func, root, type in self.history]
        }
    
    def create_midi_file(self, progression, filename="chord_progression.mid", tempo=90, 
                         instrument=0, melody_instrument=73, volume=100, melody_volume=100,
                         time_signature=(3, 4), include_melody=True):
        """
        Create a MIDI file from a chord progression with optional melody.
        
        Args:
            progression: List of chord dictionaries.
            filename: Output MIDI filename.
            tempo: Tempo in BPM.
            instrument: MIDI instrument number for chords (0-127, default 0=Acoustic Grand Piano).
            melody_instrument: MIDI instrument number for melody (0-127, default 73=Flute).
            volume: MIDI velocity for chords (0-127).
            melody_volume: MIDI velocity for melody (0-127).
            time_signature: Tuple of (numerator, denominator) for time signature.
            include_melody: Whether to include a generated melody.
        """
        # Create a MIDI file with 2 tracks (chords and melody)
        track_count = 2 if include_melody else 1
        midi = MIDIFile(track_count)
        
        # Track numbers
        chord_track = 0
        melody_track = 1
        
        # Initial time position
        time_position = 0
        
        # Add track name and tempo
        midi.addTrackName(chord_track, time_position, "Chord Progression")
        midi.addTempo(chord_track, time_position, tempo)
        
        if include_melody:
            midi.addTrackName(melody_track, time_position, "Melody")
            midi.addTempo(melody_track, time_position, tempo)
        
        # Set time signature (numerator/denominator)
        numerator, denominator = time_signature
        
        # Calculate duration of one measure in beats
        measure_duration = numerator * (4 / denominator)  # In quarter notes
        
        # Duration of eighth note in beats (quarter notes)
        eighth_note_duration = 0.5
        
        # Create a melody generator if needed
        if include_melody:
            melody_gen = MelodyGenerator(key=self.key)
        
        # Add each chord to the MIDI file
        for chord_item in progression:
            # Set chord instrument
            midi.addProgramChange(chord_track, 0, time_position, instrument)
            
            # Sort the notes from low to high for arpeggiation
            sorted_notes = sorted(chord_item["voicing"])
            
            # Add each note in the chord as an arpeggiated eighth note
            current_position = time_position
            for note in sorted_notes:
                # Add note (track, channel, pitch, time, duration, volume)
                midi.addNote(chord_track, 0, note, current_position, eighth_note_duration, volume)
                current_position += eighth_note_duration
            
            # Calculate remaining time in the measure for rest
            # Each chord takes len(sorted_notes) eighth notes
            notes_duration = len(sorted_notes) * eighth_note_duration
            rest_duration = measure_duration - notes_duration
            
            # Generate and add melody if requested
            if include_melody:
                # Set melody instrument
                midi.addProgramChange(melody_track, 0, time_position, melody_instrument)
                
                # Add chord notes to the chord_item for melody generation
                if "notes" not in chord_item:
                    chord_item["notes"] = [note % 12 for note in sorted_notes]
                
                # Generate melody for this chord
                melody_notes = melody_gen.generate_melody_for_chord(chord_item, measure_duration)
                
                # Add melody notes to the MIDI file
                for note, start_time, duration in melody_notes:
                    if note > 0:  # Skip rests (represented by note <= 0)
                        note_position = time_position + start_time
                        midi.addNote(melody_track, 0, note, note_position, duration, melody_volume)
            
            # Move to next measure
            time_position += measure_duration
        
        # Write the MIDI file
        with open(filename, "wb") as output_file:
            midi.writeFile(output_file)
        
        print(f"MIDI file created: {filename} in {time_signature[0]}/{time_signature[1]} time")
        return filename

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
        
        # Rhythmic patterns (in eighth notes)
        # Note: 0 represents a rest
        self.rhythmic_patterns = [
            [1, 1, 0, 1, 0, 1],         # Eighth, eighth, rest, eighth, rest, eighth
            [2, 0, 1, 0, 1],            # Quarter, rest, eighth, rest, eighth
            [1, 0, 1, 0, 2],            # Eighth, rest, eighth, rest, quarter
            [2, 0, 2],                  # Quarter, rest, quarter
            [1, 0.5, 0.5, 0, 1, 0, 1],  # Eighth, two sixteenths, rest, eighth, rest, eighth
            [3, 0, 1],                  # Dotted quarter, rest, eighth
            [1, 0, 3],                  # Eighth, rest, dotted quarter
            [2, 0, 0, 2],               # Quarter, two rests, quarter
            [0, 1, 1, 0, 1, 0, 1],      # Rest, eighth, eighth, rest, eighth, rest, eighth
            [0, 0, 4],                  # Two rests, half note
        ]
        
        if seed is not None:
            random.seed(seed)
    
    def _build_scale(self, key, scale_type):
        """Build a scale based on key and scale type."""
        pattern = self.scale_patterns.get(scale_type, self.scale_patterns["major"])
        return [(key + interval) % 12 for interval in pattern]
    
    def _is_chord_tone(self, note, chord_notes):
        """Check if a note is a chord tone."""
        return (note % 12) in chord_notes
    
    def _get_chord_scale(self, chord):
        """Get appropriate scale for a given chord."""
        # Extract chord information
        root = chord["root"]
        chord_type = chord["type"]
        
        # Choose scale based on chord type
        if chord_type == ChordType.MAJOR or chord_type == ChordType.MAJOR7:
            return self._build_scale(root, "major")
        elif chord_type == ChordType.MINOR or chord_type == ChordType.MINOR7:
            return self._build_scale(root, "minor")
        elif chord_type == ChordType.DIMINISHED or chord_type == ChordType.HALF_DIMINISHED:
            return self._build_scale((root + 3) % 12, "harmonic_minor")  # Related harmonic minor
        elif chord_type == ChordType.DOMINANT7:
            # For dominant chords, use mixolydian mode (major with flat 7)
            major_scale = self._build_scale(root, "major")
            major_scale[6] = (major_scale[6] - 1) % 12  # Flatten the 7th
            return major_scale
        else:
            # Default to major scale
            return self._build_scale(root, "major")
    
    def _get_next_note_options(self, chord, prev_note, scale):
        """Get possible next notes based on current chord and previous note."""
        options = []
        
        # Define movement options (in semitones)
        step_options = [-2, -1, 1, 2]  # Small steps
        leap_options = [-5, -4, -3, 3, 4, 5]  # Medium leaps
        large_leap_options = [-12, -8, -7, 7, 8, 12]  # Large leaps (use sparingly)
        
        # Add step options (higher probability)
        for step in step_options:
            candidate = prev_note + step
            if self.lower_bound <= candidate <= self.upper_bound:
                # Check if note is in scale
                if (candidate % 12) in scale:
                    # Higher weight for chord tones
                    weight = 5 if self._is_chord_tone(candidate, chord["notes"]) else 3
                    options.append((candidate, weight))
        
        # Add leap options (medium probability)
        for leap in leap_options:
            candidate = prev_note + leap
            if self.lower_bound <= candidate <= self.upper_bound:
                if (candidate % 12) in scale:
                    # Medium weight for leaps
                    weight = 3 if self._is_chord_tone(candidate, chord["notes"]) else 1
                    options.append((candidate, weight))
        
        # Add large leap options (low probability)
        # Only add if we haven't had a large leap recently (simplified version)
        if random.random() < 0.15:  # 15% chance to consider large leaps
            for leap in large_leap_options:
                candidate = prev_note + leap
                if self.lower_bound <= candidate <= self.upper_bound:
                    if (candidate % 12) in scale:
                        # Low weight for large leaps
                        weight = 1 if self._is_chord_tone(candidate, chord["notes"]) else 0.5
                        options.append((candidate, weight))
        
        # If no options found (rare), add the chord root in an appropriate octave
        if not options:
            root_note = chord["root"]
            # Find the closest octave of the root to the previous note
            octave = (prev_note // 12)
            candidate = root_note + (octave * 12)
            
            # Adjust if outside range
            if candidate < self.lower_bound:
                candidate += 12
            elif candidate > self.upper_bound:
                candidate -= 12
                
            options.append((candidate, 5))
        
        return options
    
    def _choose_next_note(self, options):
        """Choose the next note based on weighted options."""
        notes, weights = zip(*options)
        return random.choices(notes, weights=weights, k=1)[0]
    
    def _get_rhythmic_pattern(self, measure_duration):
        """Get a rhythmic pattern that fits within the measure duration."""
        # Choose a pattern that fits the measure
        valid_patterns = []
        for pattern in self.rhythmic_patterns:
            if sum(pattern) <= measure_duration:
                valid_patterns.append(pattern)
        
        if not valid_patterns:
            # Fallback to simple pattern if none fit
            return [1, 0, 1, 0, 1, 0]  # Simple pattern with rests
        
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
        melody_notes = []
        
        # Get the appropriate scale for this chord
        scale = self._get_chord_scale(chord)
        
        # Start with the current position
        current_note = self.current_position
        
        # Get a rhythmic pattern
        rhythm = self._get_rhythmic_pattern(measure_duration)
        
        # Convert rhythm to absolute positions
        current_time = 0
        for note_duration in rhythm:
            if note_duration > 0:  # This is a note, not a rest
                # Get options for the next note
                options = self._get_next_note_options(chord, current_note, scale)
                
                # Choose the next note
                next_note = self._choose_next_note(options)
                
                # Add the note to the melody
                melody_notes.append((next_note, current_time, note_duration))
                
                # Update current note
                current_note = next_note
            else:
                # This is a rest, represented by a negative note value
                melody_notes.append((-1, current_time, abs(note_duration)))
            
            # Update time position
            current_time += abs(note_duration)
        
        # Update the current position for the next chord
        self.current_position = current_note
        
        return melody_notes

# Example usage:
if __name__ == "__main__":
    # Create a chord generator in C minor
    chord_gen = TymoczkoChordGenerator(key=0)
    chord_gen.current_function = "i"  # Start with minor tonic
    
    print(f"Starting in key: {NOTE_NAMES[chord_gen.key]} minor")
    
    # Generate and print a progression of 16 chords
    print("\nGenerating chord progression:")
    print("-" * 50)
    
    try:
        # Store the progression
        chord_sequence = []
        
        for chord_num in range(16):
            chord = chord_gen.get_next_chord()
            chord_name = chord_gen.get_chord_name()
            function = chord["function"]
            voicing = [f"{NOTE_NAMES[n%12]}{n//12-1}" for n in chord["voicing"]]
            
            print(f"Chord {chord_num+1}: {chord_name} ({function}) - Voicing: {', '.join(voicing)}")
            
            # Add to progression
            chord_sequence.append(chord)
        
        # Print final state
        print("\nFinal state:")
        print(chord_gen.get_current_state())
        
        # MIDI instrument options
        # 0: Acoustic Grand Piano
        # 4: Electric Piano
        # 24: Acoustic Guitar (nylon)
        # 32: Acoustic Bass
        # 48: String Ensemble
        
        # Create MIDI file with piano sound
        midi_file = chord_gen.create_midi_file(
            chord_sequence, 
            filename="chord_progression.mid", 
            tempo=80,
            instrument=0,  # Acoustic Grand Piano
            melody_instrument=73,  # Flute
            volume=100,
            melody_volume=100,
            include_melody=True
        )
        
        # Attempt to play the MIDI file with the system's default player
        print(f"\nAttempting to open {midi_file} with the default MIDI player...")
        try:
            if os.name == 'posix':  # macOS or Linux
                subprocess.run(["open", midi_file], check=True)
            else:
                print("Unsupported operating system. Please open the MIDI file manually.")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Could not open MIDI file automatically: {e}")
            print(f"Please open {midi_file} manually with your MIDI player.")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")