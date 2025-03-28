#!/bin/bash

# Kill any existing FluidSynth processes
echo "Stopping any existing FluidSynth processes..."
pkill -f fluidsynth

# Wait a moment to ensure processes are terminated
sleep 1

# Start FluidSynth in a new terminal window with reverb settings
echo "Starting FluidSynth in a new terminal window with reverb..."
osascript -e 'tell application "Terminal" to do script "fluidsynth -a coreaudio -m coremidi -r 44100 -R 1 -C 1 -g 1.0 -o synth.reverb.active=yes -o synth.reverb.room-size=0.8 -o synth.reverb.width=0.9 -o synth.reverb.damp=0.4 /Users/wdickerson/Repos/scratchpad/gs/gs.sf2"'

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
  python infinite_spa.py --infinite --no-open
else
  python infinite_spa.py --no-open
fi

echo ""
echo "FluidSynth is still running in a separate terminal window."
echo "Close that terminal window when you're done listening to stop FluidSynth."
