#!/bin/bash
# Simple installation script for MorphoFeatures

echo "Installing MorphoFeatures in development mode..."
pip install -e .

echo "Creating a symlink to make imports work from any directory..."
python -c "import site; print(site.getsitepackages()[0])" | xargs -I{} ln -sf $(pwd)/morphofeatures {}/morphofeatures

echo "Installation complete. You can now import MorphoFeatures from any directory."
echo "For example: from morphofeatures import TextureNet" 