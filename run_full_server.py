#!/usr/bin/env python3
"""
Direct script to run the full integrated server
"""
import os
import sys
import subprocess

# Change to the project directory
project_path = '/Users/hyoseop1231/AI_Coding/ppuri-projects/ppuri_ai_ultimate'
os.chdir(project_path)

print(f"🚀 Starting PPuRI-AI Ultimate Full Integrated Server")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🐍 Python executable: {sys.executable}")
print("=" * 60)

# Run the full integrated server
try:
    subprocess.run([sys.executable, "full_integrated_server.py"], cwd=project_path)
except KeyboardInterrupt:
    print("\n⏹️  Server stopped by user")
except Exception as e:
    print(f"❌ Error running server: {e}")