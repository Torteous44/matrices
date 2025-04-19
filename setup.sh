#!/bin/bash
# Setup script for 3D Reconstruction

# Stop on error
set -e

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify activation
echo "Using Python: $(which python3)"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download sample data (optional, uncomment if needed)
# echo "Downloading sample data..."
# mkdir -p data/sample
# wget -P data/sample https://example.com/sample_images.zip
# unzip data/sample/sample_images.zip -d data/sample
# rm data/sample/sample_images.zip

echo "Setup complete!"
echo "To run the pipeline:"
echo "  source .venv/bin/activate  # if not already activated"
echo "  python3 scripts/run_pipeline.py --images data/sample --visualise" 