#!/bin/bash

# Kill any existing FluidSynth processes
echo "Stopping any existing FluidSynth processes..."
pkill -f fluidsynth

# Wait a moment to ensure processes are terminated
sleep 1

# Start FluidSynth in a new terminal window
echo "Starting FluidSynth in a new terminal window..."
osascript -e 'tell application "Terminal" to do script "fluidsynth -a coreaudio -m coremidi /Users/wdickerson/Repos/scratchpad/gs/gs.sf2"'

# Wait for FluidSynth to initialize
echo "Waiting for FluidSynth to initialize..."
sleep 3

# Activate virtual environment and run the program
echo "Running Infinite Spa Music Generator..."
cd "$(dirname "$0")"
source venv/bin/activate
python infinite_spa.py

echo ""
echo "FluidSynth is still running in a separate terminal window."
echo "Close that terminal window when you're done listening to stop FluidSynth."
