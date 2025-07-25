#!/bin/bash

# Audiophile Setup Script
# Automates the initial setup process

set -e  # Exit on any error

echo "🎵 Audiophile Setup Script"
echo "=========================="

# Check Python version
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✅ Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp config/env_template.txt .env
    echo "📝 Please edit .env file with your Spotify credentials"
else
    echo "✅ .env file already exists"
fi

# Create data directory
mkdir -p data logs

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Get Spotify API credentials from: https://developer.spotify.com/dashboard"
echo "2. Edit the .env file with your credentials:"
echo "   - SPOTIFY_CLIENT_ID"
echo "   - SPOTIFY_CLIENT_SECRET"
echo "3. Run the application:"
echo "   source .venv/bin/activate"
echo "   python src/main.py"
echo ""
echo "💡 Tip: You can test authentication first with:"
echo "   python src/auth/spotify_auth.py"
