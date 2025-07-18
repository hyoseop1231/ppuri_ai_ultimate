#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Set working directory
os.chdir('/Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate')

# Add project path to Python path
sys.path.insert(0, '/Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate')

# Execute the server
exec(open('simple_stable_server.py').read())