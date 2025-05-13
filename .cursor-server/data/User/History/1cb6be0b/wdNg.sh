#!/bin/bash
set -e

echo "=== Setting up environment ==="

# Fix scipy and scikit-image versions
pip install --force-reinstall scipy==1.8.0 scikit-image==0.19.3

# Fix torch-cluster and other PyTorch geometric extensions
echo "Installing/fixing PyTorch Geometric dependencies..."
pip uninstall -y torch-scatter torch-sparse torch-cluster torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# Install optional dependencies (with error handling)
echo "Installing optional dependencies..."
pip install wandb || echo "Wandb installation failed, continuing without it"

# Install neurofire and inferno-pytorch if texture models are needed
if [ "$1" = "texture" ] || [ "$1" = "all" ]; then
  echo "Installing texture model dependencies..."
  pip install neurofire inferno-pytorch || echo "Texture model dependencies installation failed"
fi

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

# Set up texture directories if needed
if [ "$1" = "texture" ] || [ "$1" = "all" ]; then
  mkdir -p experiments/lowres_texture_model/Weights
  mkdir -p experiments/lowres_texture_model/Logs
  mkdir -p experiments/highres_texture_model/Weights
  mkdir -p experiments/highres_texture_model/Logs
fi

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

# Choose what to run based on argument
if [ "$1" = "texture" ]; then
  # Run lowres texture model
  echo "=== Running lowres texture model training ==="
  python train_morphofeatures_models.py --model lowres --config configs/lowres_texture_config.yaml
  
  # Run highres texture model
  echo "=== Running highres texture model training ==="
  python train_morphofeatures_models.py --model highres --config configs/highres_texture_config.yaml
  
  echo "=== Texture model training complete ==="
elif [ "$1" = "all" ]; then
  # Run all models
  echo "=== Running all models ==="
  python train_morphofeatures_models.py --model all --config configs
  
  echo "=== All model training complete ==="
else
  # Default: run just the shape model
  echo "=== Running shape model training only ==="
  python train_morphofeatures_models.py --model shape --config configs/shape_config.yaml
  
  echo "=== Shape model training complete ==="
fi 