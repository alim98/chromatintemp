#!/bin/bash
set -e

echo "=== Setting up environment ==="

# Fix scipy and scikit-image versions
pip install --force-reinstall scipy==1.8.0 scikit-image==0.19.3

# Install optional dependencies (with error handling)
echo "Installing optional dependencies..."
pip install wandb || echo "Wandb installation failed, continuing without it"

echo "=== Environment setup complete ==="
echo ""

# Check for config files
if [ ! -d "configs" ]; then
  echo "Creating configs directory..."
  mkdir -p configs
fi

# Create experiment directories if they don't exist
mkdir -p data/mesh_cache
mkdir -p experiments/shape_model/checkpoints
mkdir -p experiments/shape_model/logs
mkdir -p experiments/lowres_texture_model/Weights
mkdir -p experiments/lowres_texture_model/Logs
mkdir -p experiments/highres_texture_model/Weights
mkdir -p experiments/highres_texture_model/Logs

# Check if data directory exists
if [ ! -d "nuclei_sample_1a_v1" ]; then
  echo "Warning: The data directory 'nuclei_sample_1a_v1' does not exist. Please make sure your data is available."
  exit 1
fi

# Check if metadata file exists
if [ ! -f "chromatin_classes_and_samples.csv" ]; then
  echo "Warning: The metadata file 'chromatin_classes_and_samples.csv' does not exist. Please make sure your metadata is available."
  exit 1
fi

echo "=== Using data from nuclei_sample_1a_v1 with metadata from chromatin_classes_and_samples.csv ==="

# Training parameters
echo "=== Running training script ==="
echo "Training all models using the configs in the configs directory..."
python train_morphofeatures_models.py --model all --config configs

echo "=== Training complete ===" 