# Chromatin Analysis Project

This project provides tools for analyzing and classifying 3D nuclei samples using deep learning.

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- Git (for cloning this repository)

### Creating the Environment
1. Clone this repository:
   ```bash
   git clone https://github.com/alim98/Chromatin
   cd Chromatin
   ```

2. Create the conda environment from the YAML file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate chromatin
   ```

## Project Structure
- `data/`: Contains nuclei sample data
- `dataloader/`: Data loading utilities
- `model/`: Neural network implementation
- `scripts/`: Utility scripts for data processing and visualization
- `results/`: Output directory for analysis results and visualizations

## Usage

### Fine-tuning VGG3D Model
```bash
python scripts/finetune_vgg3d.py --data_dir data/nuclei_sample_1a_v1 --batch_size 4 --epochs 20
```

### Visualizing Nuclei Samples
```bash
python scripts/visualize_example.py --mode 2d --num_samples 5
```

### Creating Nuclei Index
```bash
python scripts/create_nuclei_index.py --data_dir data/nuclei_sample_1a_v1
```

### Working with 3D Meshes
This project supports generating triangle meshes from nuclei mask volumes using the marching cubes algorithm.

#### Visualizing Meshes
```bash
python scripts/visualize_mesh.py --sample_id 1 --smooth 2
```

#### Comparing Point Clouds and Meshes
To visualize both point cloud and mesh representations side by side:
```bash
python scripts/visualize_mesh.py --sample_id 1 --compare
```

#### Using Meshes in the Dataloader
You can switch between point clouds and meshes in your code:
```python
from dataloader.mesh_dataloader import get_mesh_dataloader

# For mesh data
mesh_loader = get_mesh_dataloader(
    root_dir="data/nuclei_sample_1a_v1",
    class_csv_path="chromatin_classes_and_samples.csv",
    use_mesh=True,  # Set to True for meshes, False for point clouds
    smoothing_iterations=1,  # Control mesh smoothness
    cache_dir="data/pointclouds_cache"
)

# Access the mesh data
for batch in mesh_loader:
    vertices = batch['vertices']  # Shape: [batch_size, max_vertices, 3]
    faces = batch['faces']        # Shape: [batch_size, max_faces, 3]
    vertex_masks = batch['vertex_masks']  # Masks for valid vertices
    face_masks = batch['face_masks']      # Masks for valid faces
    break
```

Meshes provide a more complete surface representation compared to point clouds, which can be beneficial for visualization and analysis of complex chromatin structures.

# Chromatin MorphoFeatures Training

This repository contains tools to train different MorphoFeatures models for chromatin analysis.

## Setup

1. Make sure you have the required dependencies installed:
   ```
   source chromatin_py311_env/bin/activate
   ```

2. For texture model training, additional dependencies are required:
   ```bash
   # Install inferno-pytorch
   python -m pip install inferno-pytorch
   
   # Clone and install neurofire from source
   git clone https://github.com/inferno-pytorch/neurofire.git
   cd neurofire
   pip install .
   cd ..
   ```
   Note: If texture dependencies are not installed, the script will still run for shape models but will skip texture model training.

3. Configuration files are located in the `configs` directory:
   - `shape_config.yaml`: Configuration for the shape model
   - `lowres_texture_config.yaml`: Configuration for the low-resolution texture model
   - `highres_texture_config.yaml`: Configuration for the high-resolution texture model

## Training Models

### Train Shape Model Only

To train only the shape model:

```bash
python train_morphofeatures_models.py --model shape --config configs/shape_config.yaml
```

### Train Low-Resolution Texture Model Only

To train only the low-resolution texture model:

```bash
python train_morphofeatures_models.py --model lowres --config configs/lowres_texture_config.yaml
```

### Train High-Resolution Texture Model Only

To train only the high-resolution texture model:

```bash
python train_morphofeatures_models.py --model highres --config configs/highres_texture_config.yaml
```

### Train All Models

To train all models in sequence (shape, low-res, high-res):

```bash
python train_morphofeatures_models.py --model all --config configs
```

## Model Information

### Shape Model
The shape model uses a Deep Graph Convolutional Network (DeepGCN) to learn 3D shape features from point cloud data. It employs contrastive learning to differentiate between different chromatin structures.

### Texture Models
- **Low-resolution texture model**: A 3D UNet that learns coarse texture features from volumetric data.
- **High-resolution texture model**: A 3D UNet with more capacity that captures fine-grained texture details.

## Expected Data Format

- **Shape data**: Point clouds representing mesh vertices and feature information
- **Texture data**: 3D volumetric data representing chromatin density

## Outputs

Training outputs are saved in the following directories:
- Shape model: `experiments/shape_model/`
- Low-res texture model: `experiments/lowres_texture_model/`
- High-res texture model: `experiments/highres_texture_model/`

Each directory contains:
- `checkpoints/`: Model weights
- `Logs/`: Training logs and metrics (viewable with TensorBoard)
