import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import glob
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader
from scripts.visualize import NucleiVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize nuclei data from dataloader')
    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT, 
                        help='Path to nuclei dataset')
    parser.add_argument('--class_csv', type=str, default=config.CLASS_CSV_PATH,
                        help='Path to class CSV file')
    parser.add_argument('--output_dir', type=str, default=config.VISUALIZATION_OUTPUT_DIR,
                        help='Directory to save visualizations')
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='2d',
                        help='Visualization mode: 2d (slices) or 3d (volumes)')
    parser.add_argument('--class_id', type=int, nargs='+', default=None,
                        help='Filter by class ID(s). If not specified, all classes will be included.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for the dataloader')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--slice_range', type=int, nargs=2, default=None,
                        help='Range of slice numbers to include (for 2D mode only)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualizations (in addition to saving)')
    parser.add_argument('--max_crops', type=int, default=8,
                        help='Maximum number of crops per volume')
    parser.add_argument('--target_size', type=int, nargs=3, default=[80, 80, 80],
                        help='Target size for volumes (depth, height, width) for deep learning models')
    parser.add_argument('--no_resize', action='store_true',
                        help='Disable resizing to target size (use original size)')
    parser.add_argument('--compare_resize', action='store_true',
                        help='Show side-by-side comparison of original and resized volumes')
    
    return parser.parse_args()


def load_original_volume(root_dir, sample_id):
    """
    Load the original 3D volume and its mask directly from the dataset without resizing.
    
    Args:
        root_dir (str): Root directory of the dataset.
        sample_id (str): ID of the sample to load.
        
    Returns:
        tuple: (volume, mask) as numpy arrays, or (None, None) if loading fails.
    """
    sample_path = os.path.join(root_dir, sample_id)
    raw_dir = os.path.join(sample_path, 'raw')
    mask_dir = os.path.join(sample_path, 'mask')
    
    # Check if directories exist
    if not (os.path.exists(raw_dir) and os.path.exists(mask_dir)):
        print(f"Sample directories not found for {sample_id}")
        return None, None
    
    # Find all image files
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
    
    if not raw_files:
        print(f"No image files found for sample {sample_id}")
        return None, None
    
    # Load the first image to get dimensions
    first_img = np.array(Image.open(raw_files[0]))
    height, width = first_img.shape
    depth = len(raw_files)
    
    # Initialize volume arrays
    volume = np.zeros((depth, height, width), dtype=np.float32)
    mask_volume = np.zeros((depth, height, width), dtype=np.float32)
    
    # Load each slice
    for i, raw_file in enumerate(raw_files):
        # Get corresponding mask file
        file_name = os.path.basename(raw_file)
        mask_file = os.path.join(mask_dir, file_name)
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_file):
            continue
        
        # Load raw and mask images
        raw_img = np.array(Image.open(raw_file))
        mask_img = np.array(Image.open(mask_file))
        
        # Add to volume
        volume[i] = raw_img
        mask_volume[i] = mask_img
    
    print(f"Loaded original volume with shape {volume.shape}")
    return volume, mask_volume


def main():
    args = parse_args()
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor()
        # Removed normalization for visualization
    ])
    
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataloader based on mode
    if args.mode == '2d':
        # 2D slices mode
        dataloader = get_nuclei_dataloader(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            transform=transform,
            mask_transform=mask_transform,
            class_csv_path=args.class_csv,
            filter_by_class=args.class_id,
            return_paths=True,
            load_volumes=False
        )
        
        print(f"Created 2D dataloader with {len(dataloader.dataset)} samples")
        
    else:
        # 3D volumes mode
        # Use target_size only if --no_resize is not set
        target_size = tuple(args.target_size) if not args.no_resize else None
        
        dataloader = get_nuclei_dataloader(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            transform=transform,
            mask_transform=mask_transform,
            class_csv_path=args.class_csv,
            filter_by_class=args.class_id,
            return_paths=True,
            load_volumes=True,
            target_size=target_size,
            crop_size=(80, 80, 80)
        )
        
        print(f"Created 3D dataloader with {len(dataloader.dataset)} samples")
        if target_size:
            print(f"Volumes will be resized to {target_size}")
        else:
            print(f"Using original volume sizes (no resizing)")
    
    # Create visualizer
    visualizer = NucleiVisualizer(output_dir=args.output_dir, cmap='gray')
    
    # Visualize samples
    num_visualized = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        
        if args.mode == '2d':
            # Visualize 2D slices
            visualizer.visualize_batch(batch, max_samples=args.batch_size, show=args.show)
            
        elif args.mode == '3d':
            # Visualize 3D volumes
            print(f"Processing 3D volumes, batch size: {len(batch['volume'])}")
            
            for i in range(min(args.batch_size, len(batch['volume']))):
                # Extract the i-th sample
                sample = {}
                for key in batch.keys():
                    if key == 'metadata':
                        sample[key] = {k: batch[key][k][i] for k in batch[key]}
                    else:
                        sample[key] = batch[key][i]
                
                # Get sample ID for file naming
                sample_id = sample['metadata'].get('sample_id', f'sample_{i}')
                print(f"Processing sample {i+1}/{min(args.batch_size, len(batch['volume']))}, ID: {sample_id}")
                
                # If compare_resize is enabled and we are using resizing
                if args.compare_resize and not args.no_resize:
                    print(f"Comparing resize for sample: {sample_id}")
                    # We need to load the original (unresized) volume
                    original_volume, original_mask = load_original_volume(args.data_dir, sample_id)
                    
                    if original_volume is not None:
                        print(f"Original volume shape: {original_volume.shape}")
                        print(f"Resized volume shape: {sample['volume'].shape}")
                        
                        # Create side-by-side comparison static image
                        save_path = os.path.join(args.output_dir, f"{sample_id}_resize_comparison.png")
                        print(f"Saving resize comparison image to: {save_path}")
                        visualizer.visualize_resize_comparison(
                            sample, 
                            original_volume, 
                            original_mask, 
                            save_path=save_path,
                            show=args.show
                        )
                        
                        # Create side-by-side comparison animation
                        save_path = os.path.join(args.output_dir, f"{sample_id}_resize_comparison_animation.gif")
                        print(f"Saving resize comparison animation to: {save_path}")
                        visualizer.visualize_resize_comparison_animation(
                            sample,
                            original_volume,
                            original_mask,
                            save_path=save_path,
                            show=args.show,
                            axis='z',
                            frames=15,
                            interval=250
                        )
                    else:
                        print(f"Failed to load original volume for sample {sample_id}")
                
                # Standard visualizations (always performed)
                # Visualize middle slice from volume
                visualizer.visualize_slice(sample, show=args.show)
                
                # Create animation for z-axis slicing
                save_path = os.path.join(args.output_dir, f"{sample_id}_volume_animation.gif")
                visualizer.visualize_volume(sample, save_path=save_path, show=args.show,
                                            axis='z', frames=20)
                    
        # Count the number of samples visualized in this batch
        batch_size = len(batch['volume' if 'volume' in batch else 'image'])
        num_visualized += batch_size
        
        # Break if we've visualized enough samples
        if num_visualized >= args.num_samples:
            break
    
    print(f"Finished visualizing {num_visualized} samples. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 