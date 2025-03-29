"""
Generator module for the Infinite Spa Music Generator.

Contains the VoiceLeadingChordGenerator class for generating
harmonically sensible chord progressions with voice-leading constraints.
"""

import random
try:
    import numpy as np
except ImportError:
    pass  # Dependencies will be checked in main.py

from infinite_spa.harmony import (
    ChordType, NOTE_NAMES, CHORD_STRUCTURES, FUNCTION_MAP, 
    VOICE_LEADING_GRAPH, VOICE_LEADING_WEIGHTS, get_chord_name
)

class VoiceLeadingChordGenerator:
    """
    Generate a chord progression with voice-leading constraints.
    
    Args:
        key: Integer representing the key (0=C, 1=C#, etc.)
        seed: Random seed for reproducibility
    """
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
            
        return get_chord_name(chord_to_name["root"], chord_to_name["type"])
    
    def get_current_state(self):
        """Get the current state of the generator."""
        return {
            "key": NOTE_NAMES[self.key],
            "current_function": self.current_function,
            "current_chord": self.get_chord_name(),
            "voicing": self.voicing,
            "history": [(func, NOTE_NAMES[root], type.value) for func, root, type in self.history]
        }
