"""
MIDI utilities module for the Infinite Spa Music Generator.

Contains functions for MIDI file creation and playback.
"""

import os
import time
import subprocess
try:
    import mido
    from midiutil import MIDIFile
except ImportError:
    pass  # Dependencies will be checked in main.py

def create_midi_file(progression, filename="chord_progression.mid", tempo=70, 
                    instrument=0, melody_instrument=73, volume=70, melody_volume=90,
                    time_signature=(3, 4), include_melody=True, open_player=True):
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
        open_player: Whether to open the MIDI file with the default player.
    """
    # Create a MIDI file with two tracks (chords and melody)
    midi_file = MIDIFile(2)
    
    # Setup tracks
    chord_track = 0
    melody_track = 1
    
    # Set track names
    midi_file.addTrackName(chord_track, 0, "Chords")
    midi_file.addTrackName(melody_track, 0, "Melody")
    
    # Set tempo and time signature
    midi_file.addTempo(chord_track, 0, tempo)
    midi_file.addTempo(melody_track, 0, tempo)
    
    # Set time signature
    numerator, denominator = time_signature
    midi_file.addTimeSignature(chord_track, 0, numerator, denominator, 24)
    
    # Set instruments
    midi_file.addProgramChange(chord_track, 0, 0, instrument)
    midi_file.addProgramChange(melody_track, 0, 0, melody_instrument)
    
    # Add chords and melody
    current_time = 0
    for chord_dict in progression:
        voicing = chord_dict["voicing"]
        
        # Add arpeggiated chord (eighth notes)
        eighth_note_duration = 0.5  # in quarter notes
        for i, note in enumerate(voicing):
            start_time = current_time + (i * eighth_note_duration)
            midi_file.addNote(chord_track, 0, note, start_time, eighth_note_duration, volume)
        
        # Add melody if requested
        if include_melody and "melody" in chord_dict:
            for note, start_offset, duration in chord_dict["melody"]:
                if note is not None:  # None represents a rest
                    midi_file.addNote(melody_track, 0, note, current_time + start_offset, 
                                     duration, melody_volume)
        
        # Move to next measure
        measure_duration = numerator  # in quarter notes
        current_time += measure_duration
    
    # Write the MIDI file
    with open(filename, "wb") as output_file:
        midi_file.writeFile(output_file)
    
    # Open with default player if requested
    if open_player:
        try:
            if os.name == 'nt':  # Windows
                try:
                    os.startfile(filename)  # Windows-specific function
                except AttributeError:
                    subprocess.run(['start', filename], shell=True, check=False)
            elif os.name == 'posix':  # macOS or Linux
                if 'darwin' in os.sys.platform:  # macOS
                    subprocess.run(['open', filename], check=False)
                else:  # Linux
                    subprocess.run(['xdg-open', filename], check=False)
        except OSError as e:
            print(f"Could not open MIDI player: {e}")

def find_fluidsynth_port():
    """
    Find the FluidSynth virtual port from the available output ports.

    Returns:
        str: The name of the FluidSynth virtual port, or None if not found.
    """
    try:
        output_ports = mido.get_output_names()
        for port in output_ports:
            if 'fluid' in port.lower() or 'synth' in port.lower():
                return port
        return None
    except ImportError:
        print("mido library not available")
        return None
    except RuntimeError as e:
        print(f"Error finding FluidSynth port: {e}")
        return None

def play_note(outport, midi_note, duration=0.5, velocity=50):
    """
    Play a MIDI note with the specified duration and velocity.

    Args:
        outport: The MIDI output port.
        midi_note: The MIDI note number to play.
        duration: The duration of the note in seconds.
        velocity: The velocity of the note.
    """
    outport.send(mido.Message('note_on', note=midi_note, velocity=velocity))
    time.sleep(duration)
    outport.send(mido.Message('note_off', note=midi_note, velocity=0))

def play_chord(outport, notes, duration=0.5, velocity=50):
    """
    Play a chord with the specified duration and velocity.

    Args:
        outport: The MIDI output port.
        notes: List of MIDI note numbers to play.
        duration: The duration of the chord in seconds.
        velocity: The velocity of the notes.
    """
    # Send note_on messages for all notes
    for note in notes:
        outport.send(mido.Message('note_on', note=note, velocity=velocity))
    
    # Wait for the specified duration
    time.sleep(duration)
    
    # Send note_off messages for all notes
    for note in notes:
        outport.send(mido.Message('note_off', note=note, velocity=0))

def play_progression_with_fluidsynth(progression, tempo=70, instrument=11,
                                    melody_instrument=73, volume=70, melody_volume=100,
                                    time_signature=(3, 4), include_melody=True):
    """
    Play a chord progression directly using FluidSynth by sending the MIDI file to FluidSynth.
    
    Args:
        progression: List of chord dictionaries.
        tempo: Tempo in BPM.
        instrument: MIDI instrument number for chords.
        melody_instrument: MIDI instrument number for melody.
        volume: MIDI velocity for chords.
        melody_volume: MIDI velocity for melody.
        time_signature: Tuple of (numerator, denominator) for time signature.
        include_melody: Whether to include a generated melody.
    
    Returns:
        bool: True if playback was successful, False otherwise.
    """
    # Create a temporary MIDI file
    temp_midi_file = "temp_progression.mid"
    create_midi_file(progression, filename=temp_midi_file, tempo=tempo, 
                    instrument=instrument, melody_instrument=melody_instrument,
                    volume=volume, melody_volume=melody_volume,
                    time_signature=time_signature, include_melody=include_melody,
                    open_player=False)
    
    # Try to find FluidSynth port
    fluidsynth_port = find_fluidsynth_port()
    
    if fluidsynth_port:
        try:
            # Open the port
            with mido.open_output(fluidsynth_port) as outport:
                # Set instruments
                outport.send(mido.Message('program_change', program=instrument, channel=0))
                outport.send(mido.Message('program_change', program=melody_instrument, channel=1))
                
                # Play the MIDI file
                for msg in mido.MidiFile(temp_midi_file).play():
                    outport.send(msg)
                    
            # Clean up temporary file
            if os.path.exists(temp_midi_file):
                os.remove(temp_midi_file)
                
            return True
        except (IOError, RuntimeError, OSError) as e:
            print(f"Error playing progression with FluidSynth: {e}")
            return False
    else:
        print("FluidSynth port not found.")
        return False
