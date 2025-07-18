#!/bin/bash

# Simple server launcher script
echo "Starting PPuRI-AI Ultimate Server..."
echo "Working directory: $(pwd)"

# Change to project directory
cd /Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate

# Try to run the server
if python3 simple_stable_server.py; then
    echo "Server started successfully"
elif python simple_stable_server.py; then
    echo "Server started with python (fallback)"
else
    echo "Failed to start server, trying direct method..."
    python3 direct_server.py
fi