#!/usr/bin/env python3
"""
Audiophile Startup Script

Simple script to start the Audiophile application with proper error handling
and dependency checking.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask',
        'flask_socketio',
        'spotipy',
        'pandas',
        'numpy',
        'scikit-learn'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True

def check_env_file():
    """Check if .env file exists with required variables."""
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ Missing .env file")
        print("Copy config/env_template.txt to .env and add your API keys")
        return False

    return True

def start_application():
    """Start the Audiophile application."""
    print("🎵 Starting Audiophile - Intelligent Music Discovery")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        return

    # Check environment
    if not check_env_file():
        return

    # Start the application
    try:
        import app
        print("✅ All checks passed. Starting application...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in a virtual environment")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Check your .env file has valid Spotify credentials")

if __name__ == "__main__":
    start_application()
