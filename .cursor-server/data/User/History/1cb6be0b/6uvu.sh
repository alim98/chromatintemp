#!/bin/bash
set -e

echo "=== Setting up environment ==="

# Fix scipy and scikit-image versions
pip install --force-reinstall scipy==1.8.0 scikit-image==0.19.3

echo "=== Environment setup complete ==="
echo ""

# Check for config files
if [ ! -d "configs" ]; then
  echo "Creating configs directory..."
  mkdir -p configs
fi

# Create sample config files if they don't exist
if [ ! -f "configs/shape_config.yaml" ]; then
  echo "Creating sample shape_config.yaml..."
  cat > configs/shape_config.yaml << EOL
# Shape Model Configuration
data:
  root_dir: data/shapes
  class_csv_path: data/metadata.csv
  num_points: 1024

loader:
  batch_size: 8
  shuffle: true
  num_workers: 0

model:
  name: PointNet
  in_channels: 3
  num_classes: 4
  feature_transform: true

training:
  loss: CrossEntropyLoss
  learning_rate: 0.001
  weight_decay: 0.0001
  num_epochs: 50
  scheduler:
    name: StepLR
    step_size: 20
    gamma: 0.5

output:
  checkpoint_dir: experiments/shape_model/checkpoints
  log_dir: experiments/shape_model/logs
  save_every: 5
EOL
fi

if [ ! -f "configs/lowres_texture_config.yaml" ]; then
  echo "Creating sample lowres_texture_config.yaml..."
  cat > configs/lowres_texture_config.yaml << EOL
# Low-res Texture Model Configuration
project_directory: experiments/lowres_texture_model
device: cuda

data_config:
  root_dir: data/textures
  class_csv_path: data/metadata.csv
  is_cytoplasm: false
  box_size: [104, 104, 104]
  split: 0.2
  seed: 42

loader_config:
  batch_size: 4
  shuffle: true
  num_workers: 0
  pin_memory: true

val_loader_config:
  batch_size: 4
  shuffle: false
  num_workers: 0
  pin_memory: true

model_name: UNet3D
model_kwargs:
  in_channels: 1
  out_channels: 1
  f_maps: [32, 64, 128, 256]
  final_sigmoid: true

loss: BCELoss
loss_kwargs: {}

training_optimizer_kwargs:
  optimizer: Adam
  optimizer_kwargs:
    lr: 0.0001
    weight_decay: 0.00001

num_epochs: 50
EOL
fi

if [ ! -f "configs/highres_texture_config.yaml" ]; then
  echo "Creating sample highres_texture_config.yaml..."
  cat > configs/highres_texture_config.yaml << EOL
# High-res Texture Model Configuration
project_directory: experiments/highres_texture_model
device: cuda

data_config:
  root_dir: data/textures_highres
  class_csv_path: data/metadata.csv
  is_cytoplasm: false
  box_size: [256, 256, 256]
  split: 0.2
  seed: 42

loader_config:
  batch_size: 2
  shuffle: true
  num_workers: 0
  pin_memory: true

val_loader_config:
  batch_size: 2
  shuffle: false
  num_workers: 0
  pin_memory: true

model_name: UNet3D
model_kwargs:
  in_channels: 1
  out_channels: 1
  f_maps: [32, 64, 128, 256]
  final_sigmoid: true

loss: BCELoss
loss_kwargs: {}

training_optimizer_kwargs:
  optimizer: Adam
  optimizer_kwargs:
    lr: 0.0001
    weight_decay: 0.00001

num_epochs: 50
EOL
fi

echo "=== Running training script ==="
echo "Training all models using the configs in the configs directory..."
python train_morphofeatures_models.py --model all --config configs

echo "=== Training complete ===" 