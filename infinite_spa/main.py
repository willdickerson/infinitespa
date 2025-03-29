"""
Main module for the Infinite Spa Music Generator.

Provides CLI interface and demo functionality for the Infinite Spa Music Generator.
"""

import os
import random
import argparse
try:
    import numpy as np
except ImportError:
    pass  # Dependencies will be checked in check_dependencies()

from infinite_spa.generator import VoiceLeadingChordGenerator
from infinite_spa.melody import MelodyGenerator
from infinite_spa.midi_utils import create_midi_file, play_progression_with_fluidsynth
from infinite_spa.harmony import NOTE_NAMES

# ASCII art for the Infinite Spa Music Generator
INFINITE_SPA_ASCII = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           ^^                   @@@@@@@@@                                     ║
║      ^^       ^^            @@@@@@@@@@@@@@@                                  ║
║                           @@@@@@@@@@@@@@@@@@              ^^                 ║
║                          @@@@@@@@@@@@@@@@@@@@                                ║
║~~~~ ~~ ~~~~~ ~~~~~~~~ ~~ &&&&&&&&&&&&&&&&&&&& ~~~~~~~ ~~~~~~~~~~~ ~~~        ║
║~         ~~   ~  ~       ~~~~~~~~~~~~~~~~~~~~ ~       ~~     ~~ ~            ║
║  ~      ~~      ~~ ~~ ~~  ~~~~~~~~~~~~~ ~~~~  ~     ~~~    ~ ~~~  ~ ~~       ║
║  ~  ~~     ~         ~      ~~~~~~  ~~ ~~~       ~~ ~ ~~  ~~ ~               ║
║~  ~       ~ ~      ~           ~~ ~~~~~~  ~      ~~  ~             ~~        ║
║      ~             ~        ~      ~      ~~   ~             ~               ║
║                                                                              ║
║                                                                              ║
║        ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩                   ║
║               ♫    I N F I N I T E    S P A    ♫                             ║
║        ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩ ♩                   ║
║                                                                              ║
║                       Procedural relaxation                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

def check_dependencies():
    """Check if required dependencies are installed and prompt user to install if missing."""
    dependencies = {
        "numpy": "numpy>=1.20.0",
        "midiutil": "midiutil>=1.2.1",
        "mido": "mido>=1.2.10",
        "rtmidi": "python-rtmidi>=1.5.0"
    }
    
    missing = []
    
    # Check each dependency
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    # Prompt to install missing dependencies
    if missing:
        print("\nMissing required dependencies:")
        for package in missing:
            print(f"  - {package}")
        
        print("\nPlease install the missing dependencies using:")
        print(f"pip install {' '.join(missing)}")
        
        user_input = input("\nWould you like to install these dependencies now? (y/n): ")
        if user_input.lower() == 'y':
            print("Installing dependencies...")
            os.system(f"pip install {' '.join(missing)}")
            print("Dependencies installed successfully.")
        else:
            print("Dependencies not installed. The program may not function correctly.")
    
    # Check if FluidSynth is installed (macOS)
    if os.name == 'posix' and 'darwin' in os.sys.platform:
        try:
            result = os.system("which fluidsynth > /dev/null 2>&1")
            if result != 0:
                print("\nFluidSynth not found. This is required for real-time audio playback.")
                print("On macOS, you can install it via Homebrew:")
                print("  brew install fluidsynth")
                
                user_input = input("\nWould you like to install FluidSynth now? (y/n): ")
                if user_input.lower() == 'y':
                    print("Installing FluidSynth via Homebrew...")
                    os.system("brew install fluidsynth")
                    print("FluidSynth installed successfully.")
                    print("Starting FluidSynth server...")
                    os.system("fluidsynth -a coreaudio -m coremidi -s &")
                    print("FluidSynth server started.")
                else:
                    print("FluidSynth not installed. Real-time audio playback will not be available.")
        except OSError as e:
            print(f"Error checking/installing FluidSynth: {e}")
            print("You may need to install FluidSynth manually for real-time audio playback.")

def generate_and_play_progression(key=None, num_chords=16, tempo=70, 
                                 instrument=8, melody_instrument=73,
                                 open_player=True, seed=None):
    """
    Generate and play a chord progression with melody.
    
    Args:
        key: Key to generate in (0=C, 1=C#, etc.). If None, a random key is chosen.
        num_chords: Number of chords to generate.
        tempo: Tempo in BPM.
        instrument: MIDI instrument for chords (0-127).
        melody_instrument: MIDI instrument for melody (0-127).
        open_player: Whether to open the MIDI file with the default player.
        seed: Random seed for reproducibility.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Choose a random key if not specified
    if key is None:
        key = random.randint(0, 11)
    
    try:
        # Create chord generator
        chord_generator = VoiceLeadingChordGenerator(key=key, seed=seed)
        
        # Generate chord progression
        progression = []
        for _ in range(num_chords):
            chord = chord_generator.get_next_chord()
            progression.append(chord)
        
        # Generate melody
        melody_generator = MelodyGenerator(key=key, seed=seed)
        for chord in progression:
            melody = melody_generator.generate_melody_for_chord(chord, 3)  # 3 quarter notes per measure
            chord["melody"] = melody
        
        # Try to play with FluidSynth first
        if not open_player:
            success = play_progression_with_fluidsynth(
                progression, tempo=tempo, instrument=instrument,
                melody_instrument=melody_instrument, include_melody=True
            )
            if success:
                return True
        
        # Fall back to MIDI file if FluidSynth not available or open_player is True
        midi_filename = "spa_music.mid"
        create_midi_file(
            progression, filename=midi_filename, tempo=tempo,
            instrument=instrument, melody_instrument=melody_instrument,
            open_player=open_player, include_melody=True
        )
        
        return True
    except (ValueError, OSError, RuntimeError) as e:
        print(f"Error generating or playing progression: {e}")
        return False

def main():
    """Main function for the Infinite Spa Music Generator."""
    # Print ASCII art
    print(INFINITE_SPA_ASCII)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Infinite Spa Music Generator")
    parser.add_argument("--key", type=int, choices=range(12), 
                        help="Key to generate in (0=C, 1=C#, etc.)")
    parser.add_argument("--chords", type=int, default=16, 
                        help="Number of chords to generate")
    parser.add_argument("--tempo", type=int, default=70, 
                        help="Tempo in BPM")
    parser.add_argument("--instrument", type=int, default=11, choices=range(128),
                        help="MIDI instrument for chords (0-127)")
    parser.add_argument("--melody-instrument", type=int, default=73, choices=range(128),
                        help="MIDI instrument for melody (0-127)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--no-open", action="store_true", 
                        help="Don't open MIDI player automatically")
    parser.add_argument("--infinite", action="store_true", 
                        help="Generate and play music infinitely (Ctrl+C to stop)")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Run in infinite mode if requested
    if args.infinite:
        print("=== INFINITE MODE ACTIVATED ===")
        print("Press Ctrl+C to stop the infinite music generation")
        try:
            while True:
                success = generate_and_play_progression(
                    key=args.key, num_chords=args.chords, tempo=args.tempo,
                    instrument=args.instrument, melody_instrument=args.melody_instrument,
                    open_player=not args.no_open, seed=args.seed
                )
                if not success:
                    print("FluidSynth not available for infinite mode. Exiting.")
                    break
                # print("\n=== Generating next progression... ===\n")
        except KeyboardInterrupt:
            print("\nInfinite mode stopped by user.")
    else:
        # Generate and play a single progression
        generate_and_play_progression(
            key=args.key, num_chords=args.chords, tempo=args.tempo,
            instrument=args.instrument, melody_instrument=args.melody_instrument,
            open_player=not args.no_open, seed=args.seed
        )

if __name__ == "__main__":
    main()
