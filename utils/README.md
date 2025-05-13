# 3D Point Cloud Generation Utilities

This directory contains utilities for generating and visualizing 3D point clouds from masked nuclei samples.

## Requirements

These utilities require the following Python packages:
- numpy
- pandas
- open3d
- pillow (PIL)
- tqdm

You can install them using pip:

```bash
pip install numpy pandas open3d pillow tqdm
```

## Files

- `pointcloud.py`: Core module with functions for loading mask volumes and generating point clouds
- `pointcloud_to_html.py`: Tool to convert point cloud PLY files to interactive 3D HTML visualizations

## Usage

### Generating Point Clouds

To generate a point cloud for a specific sample:

```bash
python -m utils.pointcloud --sample_id 0001 --data_dir data --output_dir data/pointclouds
```

To generate point clouds for all samples:

```bash
python -m utils.pointcloud --data_dir data --output_dir data/pointclouds
```

Additional options:
- `--voxel_size`: Size of each voxel in the output point cloud (default: 1.0)
- `--sample_rate`: Fraction of points to keep (default: 0.5, range: 0.0-1.0)
- `--max_points`: Maximum number of points to include in the point cloud (optional)

### Converting Point Clouds to HTML

To convert a single point cloud to an interactive HTML visualization:

```bash
python -m utils.pointcloud_to_html --input data/pointclouds/0001.ply --output data/pointclouds_html/0001.html
```

To convert all point clouds in a directory:

```bash
python -m utils.pointcloud_to_html --input data/pointclouds --output data/pointclouds_html
```

Additional options:
- `--title`: Custom title for the HTML page (default: filename)
- `--point_size`: Size of points in the visualization (default: 0.05)

### Using the Script in scripts/

There's also a script in the scripts directory that can generate point clouds for all samples and optionally create HTML visualizations:

```bash
python scripts/generate_all_pointclouds.py --data_dir data --output_dir data/pointclouds --generate_html
```

Additional options:
- `--voxel_size`: Size of each voxel in the output point cloud (default: 1.0)
- `--sample_rate`: Fraction of points to keep (default: 0.5)
- `--max_points`: Maximum number of points to include in the point cloud (optional)
- `--html_dir`: Directory to save HTML visualizations (default: data/pointclouds_html)
- `--point_size`: Size of points in the HTML visualization (default: 0.05)

## Pipeline

The point cloud generation pipeline follows these steps:

1. For each sample:
   - Load the mask volume from TIFF slices
   - Convert the binary mask to a point cloud
   - Optionally limit the number of points (using sample_rate or max_points)
   - Save the point cloud as a PLY file in the output directory

2. Optionally, convert the PLY files to interactive HTML visualizations

## Output Format

The point clouds are saved in PLY format, which can be opened with various 3D visualization software including:
- Open3D
- MeshLab
- CloudCompare
- Blender

The HTML visualizations can be opened in any modern web browser and provide interactive 3D viewing with zoom, pan, and rotate functionality. The HTML viewer also includes a slider to adjust the point size in real-time. 