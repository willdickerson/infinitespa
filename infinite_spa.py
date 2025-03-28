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
                         instrument=0, volume=100, time_signature=(3, 4)):
        """
        Create a MIDI file from a chord progression.
        
        Args:
            progression: List of chord dictionaries.
            filename: Output MIDI filename.
            tempo: Tempo in BPM.
            instrument: MIDI instrument number (0-127, default 0=Acoustic Grand Piano).
            volume: MIDI velocity (0-127).
            time_signature: Tuple of (numerator, denominator) for time signature.
        """
        # Create MIDI file with 1 track
        midi = MIDIFile(1)
        
        # Setup track
        track = 0
        time_position = 0
        
        # Add track name and tempo
        midi.addTrackName(track, time_position, "Chord Progression")
        midi.addTempo(track, time_position, tempo)
        
        # Set time signature (numerator/denominator)
        numerator, denominator = time_signature
        
        # Calculate duration of one measure in beats
        measure_duration = numerator * (4 / denominator)  # In quarter notes
        
        # Duration of eighth note in beats (quarter notes)
        eighth_note_duration = 0.5
        
        # Add each chord to the MIDI file
        for progression_idx, chord_item in enumerate(progression):
            # Set instrument
            midi.addProgramChange(track, 0, time_position, instrument)
            
            # Sort the notes from low to high for arpeggiation
            sorted_notes = sorted(chord_item["voicing"])
            
            # Add each note in the chord as an arpeggiated eighth note
            current_position = time_position
            for note in sorted_notes:
                # Add note (track, channel, pitch, time, duration, volume)
                midi.addNote(track, 0, note, current_position, eighth_note_duration, volume)
                current_position += eighth_note_duration
            
            # Calculate remaining time in the measure for rest
            # Each chord takes len(sorted_notes) eighth notes
            notes_duration = len(sorted_notes) * eighth_note_duration
            rest_duration = measure_duration - notes_duration
            
            # Move to next measure
            time_position += measure_duration
        
        # Write the MIDI file
        with open(filename, "wb") as output_file:
            midi.writeFile(output_file)
        
        print(f"MIDI file created: {filename} in {time_signature[0]}/{time_signature[1]} time")
        return filename

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
        progression = []
        
        for chord_num in range(16):
            chord = chord_gen.get_next_chord()
            chord_name = chord_gen.get_chord_name()
            function = chord["function"]
            voicing = [f"{NOTE_NAMES[n%12]}{n//12-1}" for n in chord["voicing"]]
            
            print(f"Chord {chord_num+1}: {chord_name} ({function}) - Voicing: {', '.join(voicing)}")
            
            # Add to progression
            progression.append(chord)
        
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
            progression, 
            filename="chord_progression.mid", 
            tempo=80,
            instrument=0,  # Acoustic Grand Piano
            volume=100
        )
        
        # Attempt to play the MIDI file with the system's default player
        print(f"\nAttempting to open {midi_file} with the default MIDI player...")
        try:
            if os.name == 'posix':  # macOS or Linux
                subprocess.run(["open", midi_file], check=True)
            else:
                print("Unsupported operating system. Please open the MIDI file manually.")
        except Exception as e:
            print(f"Could not open MIDI file automatically: {e}")
            print(f"Please open {midi_file} manually with your MIDI player.")
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")