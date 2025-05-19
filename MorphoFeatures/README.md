# MorphoFeatures

A modernized version of MorphoFeatures using PyTorch Lightning and MONAI instead of the outdated Inferno/Neurofire libraries.

## Installation

You can install MorphoFeatures in development mode using the provided script:

```bash
cd MorphoFeatures
chmod +x install.sh
./install.sh
```

Or manually with pip:

```bash
cd MorphoFeatures
pip install -e .
```

## Quick Start

### Training a Texture Model

Using the wrapper script:

```bash
python MorphoFeatures/train_texture_lightning.py --config MorphoFeatures/texture_lightning_config.yml --output_dir experiments/texture_model --gpus 1 --amp
```

Or using the main training script:

```bash
python train_morphofeatures_models.py --lowres_config MorphoFeatures/texture_lightning_config.yml
```

### Import in Your Code

```python
# Import the Lightning model
from morphofeatures.texture import TextureNet

# Create and use the model
model = TextureNet(config)
```

## Migration Details

Please refer to the [Migration README](migration_README.md) for details on the changes made to modernize the codebase.

## Troubleshooting

If you encounter import errors, it usually means the package is not properly installed. Try:

1. Re-running the installation script:
   ```bash
   ./install.sh
   ```

2. Making sure you can import the module in Python:
   ```python
   import morphofeatures
   print(morphofeatures.__file__)  # Should print the path to the package
   ```

3. Adding the repository to PYTHONPATH:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/MorphoFeatures
   ```
