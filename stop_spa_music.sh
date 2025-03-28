#!/bin/bash

# Script to stop all audio processes related to the spa music generator

echo "Stopping all audio processes..."

# Kill FluidSynth and afplay processes with increasing force
echo "Terminating audio processes..."
pkill -f fluidsynth
pkill -f afplay

# Wait a moment
sleep 1

# If processes are still running, force kill them
echo "Force killing any remaining audio processes..."
pkill -9 -f fluidsynth
pkill -9 -f afplay

# Force close any terminal windows running our audio loops
echo "Closing audio terminal windows..."
osascript -e 'tell application "Terminal" to close (every window whose name contains "afplay")' &> /dev/null
osascript -e 'tell application "Terminal" to close (every window whose name contains "fluidsynth")' &> /dev/null
osascript -e 'tell application "Terminal" to close (every window whose name contains "AMBIENT SOUND LOOP")' &> /dev/null

# Check if any audio processes are still running
if pgrep -f fluidsynth > /dev/null || pgrep -f afplay > /dev/null; then
    echo "WARNING: Some audio processes are still running. Trying one more approach..."
    # Try a different approach - kill by process name
    killall fluidsynth 2>/dev/null
    killall afplay 2>/dev/null
    sleep 1
    killall -9 fluidsynth 2>/dev/null
    killall -9 afplay 2>/dev/null
fi

echo "All spa music processes have been stopped."
echo "If you still hear audio, please manually close any Terminal windows playing sounds."
