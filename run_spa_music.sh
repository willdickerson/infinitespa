#!/bin/bash

# Create a cleanup function
cleanup() {
    echo "Stopping all audio processes..."
    pkill -f fluidsynth
    pkill -f afplay
    # Force close any terminal windows running our audio loops
    osascript -e 'tell application "Terminal" to close (every window whose name contains "afplay")' &> /dev/null
    osascript -e 'tell application "Terminal" to close (every window whose name contains "fluidsynth")' &> /dev/null
    osascript -e 'tell application "Terminal" to close (every window whose name contains "AMBIENT SOUND LOOP")' &> /dev/null
    # Don't exit here, let the script finish naturally
}

# Set up trap to catch Ctrl+C and other termination signals
# Only trap INT, not EXIT which causes restart loops
trap cleanup INT TERM

# Kill any existing FluidSynth processes and audio players
echo "Stopping any existing FluidSynth processes and audio players..."
pkill -f fluidsynth
pkill -f afplay  # macOS audio player
# Force close any terminal windows running our audio loops
osascript -e 'tell application "Terminal" to close (every window whose name contains "afplay")' &> /dev/null
osascript -e 'tell application "Terminal" to close (every window whose name contains "fluidsynth")' &> /dev/null
osascript -e 'tell application "Terminal" to close (every window whose name contains "AMBIENT SOUND LOOP")' &> /dev/null

# Wait a moment to ensure processes are terminated
sleep 1

# Check if ambient sound file exists
AMBIENT_SOUND="infinite_spa/ambient_sounds/babbling-brook.mp3"
USE_AMBIENT=false

if [ -f "$AMBIENT_SOUND" ]; then
    USE_AMBIENT=true
    echo "Found ambient sound file: $AMBIENT_SOUND"
else
    echo "Ambient sound file not found: $AMBIENT_SOUND"
    echo "Place an MP3 file at this location to enable ambient sounds."
fi

# Start FluidSynth in a new terminal window
echo "Starting FluidSynth in a new terminal window..."
osascript -e 'tell application "Terminal" to do script "fluidsynth -a coreaudio -m coremidi -r 44100 -R 1 -C 1 -g 1.0 -o synth.reverb.active=yes -o synth.reverb.room-size=0.95 -o synth.reverb.width=1.0 -o synth.reverb.damp=0.3 -o synth.reverb.level=0.9 -o synth.chorus.active=yes -o synth.chorus.depth=8 -o synth.chorus.speed=0.4 -o synth.chorus.level=0.6 -o synth.gain=0.9 /Users/wdickerson/Repos/scratchpad/gs/gs.sf2"'

# Start ambient sound loop if file exists
if [ "$USE_AMBIENT" = true ]; then
    echo "Starting ambient sound loop in background..."
    # Start in a new terminal with a distinctive title that we can target for closing
    osascript -e 'tell application "Terminal" to do script "echo \"AMBIENT SOUND LOOP - DO NOT CLOSE MANUALLY\"; cd \"'$(pwd)'\" && while true; do afplay \"'$AMBIENT_SOUND'\"; done"'
fi

# Wait for FluidSynth to initialize
echo "Waiting for FluidSynth to initialize..."
sleep 3

# Activate virtual environment and run the program
echo "Running Infinite Spa Music Generator..."
cd "$(dirname "$0")"
source venv/bin/activate

# Check for infinite mode flag
if [ "$1" == "--infinite" ]; then
  echo "Running in infinite mode. Press Ctrl+C to stop."
  echo "NOTE: Press Ctrl+C to stop all audio playback including the ambient sound."
  echo "If audio persists, run ./stop_spa_music.sh"
  python run_infinite_spa.py --infinite --no-open
else
  python run_infinite_spa.py --no-open
fi

# Run cleanup at the end of the script
cleanup
