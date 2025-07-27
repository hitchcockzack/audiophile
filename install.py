#!/usr/bin/env python3
"""
Audiophile Installation Script

Sets up all dependencies and configuration for the enhanced Audiophile system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Output: {e.output}")
        return False

def install_dependencies():
    """Install all required dependencies."""
    print("🚀 Installing Audiophile Dependencies")
    print("="*50)

    # Core dependencies
    dependencies = [
        "pip install --upgrade pip",
        "pip install Flask Flask-SocketIO",
        "pip install spotipy python-dotenv",
        "pip install pandas numpy scikit-learn",
        "pip install python-dateutil requests",
        "pip install librosa soundfile",  # Audio analysis
    ]

    for cmd in dependencies:
        if not run_command(cmd, f"Installing {cmd.split()[-1]}"):
            return False

    # Optional advanced dependencies
    print("\n🔬 Installing optional advanced features...")
    optional_deps = [
        "pip install textblob nltk",  # Lyrical analysis
        "pip install essentia-tensorflow",  # Advanced audio analysis
    ]

    for cmd in optional_deps:
        run_command(cmd, f"Installing {cmd.split()[-1]} (optional)")

    return True

def setup_config():
    """Set up configuration files."""
    print("\n⚙️ Setting up configuration...")

    env_template = Path("config/env_template.txt")
    env_file = Path(".env")

    if env_template.exists() and not env_file.exists():
        try:
            # Copy template to .env
            with open(env_template, 'r') as f:
                template_content = f.read()

            with open(env_file, 'w') as f:
                f.write(template_content)

            print("✅ Created .env file from template")
            print("📝 Please edit .env with your Spotify API credentials")

        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No env template found - you'll need to create .env manually")

    return True

def create_directories():
    """Create necessary directories."""
    directories = ["data", "templates"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✅ Created necessary directories")

def main():
    """Main installation process."""
    print("🎵 Audiophile Enhanced Installation")
    print("🚀 Setting up intelligent music discovery platform...")
    print("="*60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return

    print(f"✅ Python {sys.version} detected")

    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed")
        return

    # Set up configuration
    setup_config()

    # Create directories
    create_directories()

    print("\n" + "="*60)
    print("🎉 Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit .env with your Spotify API credentials")
    print("   Get them from: https://developer.spotify.com/dashboard")
    print("2. Start the application:")
    print("   python app.py")
    print("3. Navigate to: http://localhost:5001")
    print("\n🎵 Ready to discover amazing music!")

if __name__ == "__main__":
    main()
