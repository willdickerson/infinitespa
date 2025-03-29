"""
Harmony module for the Infinite Spa Music Generator.

Contains chord types, structures, functional harmony relationships,
voice-leading graphs, and weights for chord progressions.
"""

from enum import Enum

class ChordType(Enum):
    """Enum representing different chord types."""
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

# Functional harmonic relationships
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
    "ii°": {"root": 2, "type": ChordType.DIMINISHED},
    
    # Dominant functions
    "V": {"root": 7, "type": ChordType.MAJOR},
    "v": {"root": 7, "type": ChordType.MINOR},
    "V7": {"root": 7, "type": ChordType.DOMINANT7},
    "vii°": {"root": 11, "type": ChordType.DIMINISHED},
    "VII": {"root": 11, "type": ChordType.MAJOR},
}

# Voice-leading graph
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
    "ii°": ["ii°", "V", "V7", "vii°", "i", "VI", "III", "iv"],
    
    # Dominant functions typically resolve to tonic
    "V": ["V", "V7", "I", "i", "vi", "iii", "IV", "ii"],
    "V7": ["V7", "I", "i", "vi", "iii", "IV"],
    "vii°": ["vii°", "I", "i", "vi", "iii", "IV"],
    "VII": ["VII", "III", "i", "VI", "iv"],
    
    # Adding missing chord functions with complete relationships
    "III": ["III", "i", "VI", "iv", "V", "VII", "v", "ii°"],
    "VI": ["VI", "i", "III", "iv", "ii", "V", "VII"],
    "II": ["II", "V", "V7", "I", "vi", "IV", "vii°"],
    "v": ["v", "i", "VI", "III", "iv", "ii°", "VII"],
}

# Voice-leading weights (smaller = more preferred)
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

def get_chord_name(root, chord_type):
    """
    Get the formatted name of a chord.
    
    Args:
        root: Integer representing the root note (0=C, 1=C#, etc.)
        chord_type: ChordType enum value
    
    Returns:
        String representation of the chord name
    """
    root_name = NOTE_NAMES[root]
    type_str = chord_type.value
    
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
