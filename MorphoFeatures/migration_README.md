# MorphoFeatures Lightning Migration

This directory contains modernized implementations of the MorphoFeatures texture branch using PyTorch Lightning and MONAI instead of the outdated Inferno/Neurofire libraries.

## Migration Overview

The original implementation used:
- `inferno.trainers.basic.Trainer` for training loop management
- Inferno callbacks for learning rate scheduling and checkpointing
- Neurofire models for 3D UNet implementation

The new implementation uses:
- PyTorch Lightning `LightningModule` and `Trainer` for training loop management
- Lightning callbacks for learning rate scheduling and checkpointing
- MONAI networks for 3D UNet implementation

## Key Files

- `texture_lightning.py`: Lightning-based implementation of the texture model and training loop
- `predict_lightning.py`: Lightning-compatible prediction script
- `requirements_lightning.txt`: Required dependencies for the Lightning implementation

## Usage

### Installation

1. Create a new environment (optional but recommended)
   ```bash
   conda create -n morphofeatures-lightning python=3.8
   conda activate morphofeatures-lightning
   ```

2. Install the required dependencies
   ```bash
   pip install -r requirements_lightning.txt
   ```

### Training

To train the texture model:

```bash
python morphofeatures/texture/texture_lightning.py --train_config path/to/train_config.yml --data_config path/to/data_config.yml --output_dir path/to/output
```

The training configuration file (`train_config.yml`) should contain:
- `model_config`: Model configuration (feature dimensions, etc.)
- `optimizer_config`: Optimizer settings (learning rate, etc.)
- `scheduler_config`: Learning rate scheduler settings (optional)
- `num_epochs`: Number of training epochs
- `num_gpus`: Number of GPUs to use
- `use_amp`: Whether to use mixed precision training

### Prediction

To generate embeddings using a trained model:

```bash
python morphofeatures/texture/predict_lightning.py path/to/checkpoint.ckpt --config path/to/test_config.yml --output_dir path/to/output
```

For patch-based prediction:

```bash
python morphofeatures/texture/predict_lightning.py path/to/checkpoint.ckpt --config path/to/test_config_patches.yml --save_patches --output_dir path/to/output
```

To aggregate patch embeddings:

```bash
python morphofeatures/texture/predict_lightning.py path/to/checkpoint.ckpt --save_patches --aggregate_patches
```

## Benefits of Migration

- **Active maintenance & community support**: Lightning and MONAI are actively maintained and widely adopted.
- **Simpler code**: Less boilerplate code and cleaner organization.
- **Faster training**: Native DDP support and automatic mixed precision.
- **Better monitoring**: Native integration with TensorBoard and other loggers.
- **Easier extension**: Add new components or training strategies with minimal changes.

## Note on Data Loaders

The data loading logic (`CellLoaders`, `CellDataset`, etc.) has been updated to use MONAI transforms while preserving the same functionality:

- Replaced inferno transforms with MONAI equivalents:
  - `Compose` → `monai.transforms.Compose`
  - `CropPad2Size` → `SpatialPad`
  - `VolumeRandomCrop` → `RandSpatialCrop`
  - `RandomRot903D` → `RandRotate90`
  - `ElasticTransform` → `Rand3DElastic`
  - `NormalizeRange` → `ScaleIntensityRange`
  - `Cast` + `AsTorchBatch` → `EnsureType`
- Replaced `yaml2dict` with native `yaml.safe_load`

This modernization eliminates all dependencies on the outdated inferno library while maintaining the same data loading and augmentation behaviors.

## Next Steps

- The shape branch (`shape/train_shape_model.py`) could be similarly migrated to Lightning if desired.
- Consider adding Hydra for more powerful configuration management. 