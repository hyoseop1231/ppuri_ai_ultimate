#!/usr/bin/env python3
import os
import sys
import subprocess

# Change to the project directory
os.chdir('/Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate')

# Run the server
subprocess.run([sys.executable, 'simple_stable_server.py'])