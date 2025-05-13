# TIFF 3D Volume Trainer

This project provides a solution for training 3D UNet models on volumetric TIFF data, specifically designed for nuclei segmentation.

## Problem
The original training script was designed to work with `.npy` files but the actual data consists of TIFF stacks organized in directories. This solution adapts the training pipeline to work with TIFF stacks instead.

## Components

### 1. TIFF Dataloader (`tiff_dataloader.py`)
- A PyTorch dataset and dataloader for loading and processing 3D volumes from TIFF stacks
- Automatically resizes all volumes to a standardized size using scipy's zoom functionality
- Handles variable-sized input volumes to ensure compatibility with batch processing
- Supports both single sample directories and directories containing multiple samples

### 2. Training Script (`train_tiff_model.py`)
- Complete PyTorch training pipeline for 3D UNet segmentation models
- Implements a 3D UNet architecture with configurable depth and feature maps
- Provides training and validation loops with checkpointing
- Supports early stopping for faster debugging and experimentation

### 3. Configuration (`configs/lowres_texture_config.yaml`)
- YAML configuration file for easily adjusting training parameters
- Configurable model architecture, loss functions, and optimizers
- Dataset configuration including paths and target size
- Separate debug settings to speed up experimentation

## Usage

To train the model:
```bash
python train_tiff_model.py --config /path/to/config.yaml
```

## Configuration Options

Key configuration options include:
- `data_config.root_dir`: Path to the directory containing your TIFF volumes
- `data_config.box_size`: Target size for resizing all volumes (H, W, D)
- `data_config.input_dir`: Name of subdirectory containing raw TIFF slices (default: "raw")
- `data_config.target_dir`: Name of subdirectory containing mask TIFF slices (default: "mask")
- `model_kwargs`: Architecture configuration for the UNet
- `num_epochs_debug`: Number of epochs to run in debug mode (faster training)

## Requirements
- PyTorch
- tifffile
- scipy
- numpy
- PyYAML 