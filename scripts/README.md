Python scripts/finetune_vgg3d.py --sample_percent 1 --freeze_features --num_workers 0 --batch_size 1
# Nuclei Data Visualization

This folder contains scripts for visualizing nuclei data from the custom dataloader. The visualization tools support both 2D slices and 3D volumes with their corresponding masks.

## Files

- `visualize.py`: Contains the `NucleiVisualizer` class that provides methods for visualizing nuclei data
- `visualize_example.py`: An example script showing how to use the NucleiVisualizer class with the dataloader

## Usage

### Basic Usage

```python
from scripts.visualize import NucleiVisualizer
from dataloader.nuclei_dataloader import get_nuclei_dataloader

# Create a dataloader
dataloader = get_nuclei_dataloader(
    root_dir="path/to/data",
    batch_size=4,
    load_volumes=True  # Set to False for 2D slices
)

# Create visualizer
visualizer = NucleiVisualizer()

# Visualize a batch
for batch in dataloader:
    # Visualize the first sample in the batch
    sample = {k: batch[k][0] if isinstance(batch[k], (list, torch.Tensor)) else batch[k] for k in batch}
    
    # For 2D slices
    visualizer.visualize_slice(sample)
    
    # For 3D volumes
    visualizer.visualize_volume(sample, axis='z', frames=10)
    
    break  # Just visualize the first batch
```

### Running the Example Script

You can run the example script with various options:

```bash
# For 2D slice visualization
python scripts/visualize_example.py --mode 2d --show

# For 3D volume visualization
python scripts/visualize_example.py --mode 3d --show

# For visualizing multiple crops from volumes
python scripts/visualize_example.py --mode 3d --multiple_crops --show

# Filter by class ID
python scripts/visualize_example.py --class_id 1 2 3 --mode 2d --show
```

### Command Line Arguments

The example script supports the following arguments:

- `--data_dir`: Path to nuclei dataset (default: from config.DATA_ROOT)
- `--class_csv`: Path to class CSV file (default: from config.CLASS_CSV_PATH)
- `--output_dir`: Directory to save visualizations (default: from config.VISUALIZATION_OUTPUT_DIR)
- `--mode`: Visualization mode: 2d (slices), 3d (volumes), or crops (multiple crops) (default: 2d)
- `--class_id`: Filter by class ID(s)
- `--batch_size`: Batch size for the dataloader (default: 4)
- `--num_samples`: Number of samples to visualize (default: 5)
- `--slice_range`: Range of slice numbers to include for 2D mode
- `--show`: Show visualizations in addition to saving them
- `--multiple_crops`: Use multiple crops per volume
- `--max_crops`: Maximum number of crops per volume (default: 8)

## Visualization Methods

The `NucleiVisualizer` class provides the following visualization methods:

### 1. `visualize_slice(data_item, save_path=None, show=True, title=None)`

Visualizes a 2D slice from the dataset, showing the raw image, mask, and an overlay of both.

### 2. `visualize_volume(data_item, save_path=None, show=True, title=None, axis='z', frames=10, interval=200, colorbar=True)`

Creates an animation showing slices through a 3D volume along a specified axis (x, y, or z).

### 3. `visualize_multiple_crops(data_item, save_dir=None, show=True, rows=2, cols=2)`

Visualizes multiple crops from a volume in a grid layout.

### 4. `visualize_batch(batch, max_samples=4, save_dir=None, show=True)`

Visualizes multiple samples from a batch.

## Customization

You can customize the visualizations by passing additional parameters when creating the visualizer:

```python
visualizer = NucleiVisualizer(
    output_dir="custom/output/path",
    cmap='viridis',         # Colormap for raw images
    mask_cmap='hot',        # Colormap for mask overlays
    alpha=0.3,              # Alpha value for mask overlay transparency
    figsize=(12, 10)        # Figure size for plots
)
``` 