#!/usr/bin/env python3
"""
Test Runner - Direct execution of live test
"""

import sys
import os
import subprocess
import signal
import time

def run_test():
    """Run the live test directly"""
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Change to the correct directory
    os.chdir(current_dir)
    
    print("🚀 Starting Live Test")
    print("=" * 50)
    
    try:
        # Import and run the test
        from live_test import main
        main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        # Try to run as subprocess
        try:
            subprocess.run([sys.executable, "live_test.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess error: {e}")
    
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    run_test()