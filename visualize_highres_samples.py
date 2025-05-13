import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloader.highres_image_dataloader import get_highres_image_dataloader
from dataloader.highres_texture_adapter import get_morphofeatures_highres_texture_dataloader, extract_cubes

def visualize_highres_cubes(batch, output_path=None, show=True):
    """
    Visualize a batch of highres 32³ cubes (contrastive pairs).
    
    Args:
        batch: Batch from the highres texture dataloader, containing 'view1' and 'view2'
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
    """
    # Get views from batch
    view1 = batch[0]  # [B, 1, Z, H, W]
    view2 = batch[1]  # [B, 1, Z, H, W]
    
    batch_size = view1.shape[0]
    
    # Create figure for visualizing several pairs
    fig = plt.figure(figsize=(12, 4 * min(batch_size, 4)))
    
    # Show only up to 4 pairs to avoid overcrowding
    for b in range(min(batch_size, 4)):
        # Get the mid-slice for each view
        cube1 = view1[b, 0]  # [Z, H, W]
        cube2 = view2[b, 0]  # [Z, H, W]
        
        z_mid = cube1.shape[0] // 2
        slice1 = cube1[z_mid].cpu().numpy()
        slice2 = cube2[z_mid].cpu().numpy()
        
        # Normalize for visualization if needed
        if slice1.max() > 1.0:
            slice1 = slice1 / 255.0
        if slice2.max() > 1.0:
            slice2 = slice2 / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice1.min() < 0:
            slice1 = (slice1 + 1) / 2
        if slice2.min() < 0:
            slice2 = (slice2 + 1) / 2
        
        # Plot the slices side by side
        ax1 = fig.add_subplot(min(batch_size, 4), 2, b * 2 + 1)
        ax1.imshow(slice1, cmap='gray')
        ax1.set_title(f"Pair {b}, View 1")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(min(batch_size, 4), 2, b * 2 + 2)
        ax2.imshow(slice2, cmap='gray')
        ax2.set_title(f"Pair {b}, View 2")
        ax2.axis('off')
    
    plt.suptitle("Contrastive Pairs for High-Resolution Texture Learning")
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_cube_grid(cubes, output_path=None, show=True, grid_size=(4, 4)):
    """
    Visualize a grid of 32³ cubes.
    
    Args:
        cubes: List of cube tensors of shape [1, Z, H, W]
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
        grid_size: Tuple of (rows, cols) for the grid
    """
    rows, cols = grid_size
    max_cubes = rows * cols
    
    if len(cubes) < max_cubes:
        rows = int(np.ceil(len(cubes) / cols))
    
    fig = plt.figure(figsize=(cols*3, rows*3))
    
    for i, cube in enumerate(cubes[:max_cubes]):
        # Get the mid-slice
        cube_data = cube[0]  # [Z, H, W]
        z_mid = cube_data.shape[0] // 2
        slice_img = cube_data[z_mid].cpu().numpy()
        
        # Normalize for visualization if needed
        if slice_img.max() > 1.0:
            slice_img = slice_img / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice_img.min() < 0:
            slice_img = (slice_img + 1) / 2
        
        # Plot the slice
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(f"Cube {i}")
        ax.axis('off')
    
    plt.suptitle("32³ Cubes Extracted for High-Resolution Texture Model")
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Grid visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_cube_3d_slices(cube, output_path=None, show=True, slice_indices=None):
    """
    Visualize slices from a 3D cube.
    
    Args:
        cube: Tensor of shape [1, Z, H, W]
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
        slice_indices: List of z-indices to show, or None to auto-select
    """
    cube_data = cube[0]  # [Z, H, W]
    depth = cube_data.shape[0]
    
    # Select slice indices if not provided
    if slice_indices is None:
        num_slices = min(5, depth)
        slice_indices = [int(i * (depth-1) / (num_slices-1)) for i in range(num_slices)]
    
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(3*len(slice_indices), 3))
    if len(slice_indices) == 1:
        axes = [axes]
    
    for i, z_idx in enumerate(slice_indices):
        if z_idx >= depth:
            continue
            
        # Get the slice
        slice_img = cube_data[z_idx].cpu().numpy()
        
        # Normalize for visualization if needed
        if slice_img.max() > 1.0:
            slice_img = slice_img / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice_img.min() < 0:
            slice_img = (slice_img + 1) / 2
        
        # Plot the slice
        axes[i].imshow(slice_img, cmap='gray')
        axes[i].set_title(f"Z = {z_idx}")
        axes[i].axis('off')
    
    plt.suptitle("Slices from 32³ Cube")
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"3D slice visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_augmented_examples(cube, num_augmentations=5, output_path=None, show=True):
    """
    Visualize multiple augmentations of the same cube.
    
    Args:
        cube: Tensor of shape [1, Z, H, W]
        num_augmentations: Number of augmented versions to create
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
    """
    from dataloader.highres_texture_adapter import augment_cube
    
    # Create augmentations
    augmented_cubes = [cube]  # Start with the original
    for _ in range(num_augmentations):
        augmented_cubes.append(augment_cube(cube, p_rotate=0.8, p_flip=0.7, p_elastic=0.6))
    
    # Pick middle slice of each cube
    slices = []
    for c in augmented_cubes:
        z_mid = c.shape[1] // 2  # Middle slice in Z dimension
        slice_img = c[0, z_mid].cpu().numpy()
        
        # Normalize for visualization if needed
        if slice_img.max() > 1.0:
            slice_img = slice_img / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice_img.min() < 0:
            slice_img = (slice_img + 1) / 2
            
        slices.append(slice_img)
    
    # Create plot
    fig, axes = plt.subplots(1, len(slices), figsize=(3*len(slices), 3))
    
    for i, slice_img in enumerate(slices):
        title = "Original" if i == 0 else f"Aug {i}"
        axes[i].imshow(slice_img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.suptitle("Data Augmentation for Contrastive Learning")
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Augmentation visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize highres texture samples")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--sample_ids", type=str, nargs="+", help="Specific sample IDs to visualize")
    parser.add_argument("--is_cytoplasm", action="store_true", help="Use cytoplasm images instead of nucleus")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--cube_size", type=int, default=32, help="Size of cubes to extract")
    parser.add_argument("--extract_cubes", action="store_true", help="Extract and visualize individual cubes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots, just save them")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Option 1: Visualize contrastive pairs from the dataloader
    dataloader = get_morphofeatures_highres_texture_dataloader(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        class_csv_path=args.class_csv,
        sample_ids=args.sample_ids,
        is_cytoplasm=args.is_cytoplasm,
        cube_size=args.cube_size,
        debug=True
    )
    
    print(f"Created highres texture dataloader with {len(dataloader.dataset)} contrastive pairs")
    
    # Process two batches of contrastive pairs
    for i, batch in enumerate(dataloader):
        output_path = os.path.join(args.output_dir, f"highres_contrastive_batch_{i}.png") if args.output_dir else None
        visualize_highres_cubes(batch, output_path=output_path, show=not args.no_show)
        
        if i >= 1:  # Process only 2 batches
            break
    
    # Option 2: Extract and visualize individual cubes from a volume directly
    if args.extract_cubes:
        # Load the raw highres volumes
        raw_dataloader = get_highres_image_dataloader(
            root_dir=args.root_dir,
            batch_size=1,  # Process one sample at a time
            class_csv_path=args.class_csv,
            sample_ids=args.sample_ids,
            num_workers=0,
            debug=True
        )
        
        print(f"Created raw highres dataloader with {len(raw_dataloader.dataset)} samples")
        
        # Process samples
        for i, sample in enumerate(raw_dataloader):
            volume = sample['image'][0]  # [Z, 1, H, W]
            sample_id = sample['metadata']['sample_id'][0]
            
            # Extract cubes from this volume
            cubes = extract_cubes(
                volume=volume,
                cube_size=args.cube_size,
                min_foreground_percent=0.5
            )
            
            if not cubes:
                print(f"No valid cubes found for sample {sample_id}")
                continue
                
            print(f"Sample {sample_id}: Extracted {len(cubes)} valid cubes")
            
            # Visualize grid of cubes
            output_path = os.path.join(args.output_dir, f"highres_cubes_sample_{sample_id}.png") if args.output_dir else None
            visualize_cube_grid(cubes, output_path=output_path, show=not args.no_show)
            
            # Visualize 3D slices of the first cube
            if cubes:
                output_path = os.path.join(args.output_dir, f"highres_cube_slices_sample_{sample_id}.png") if args.output_dir else None
                visualize_cube_3d_slices(cubes[0], output_path=output_path, show=not args.no_show)
                
                # Visualize augmentations
                output_path = os.path.join(args.output_dir, f"highres_cube_augmentations_sample_{sample_id}.png") if args.output_dir else None
                visualize_augmented_examples(cubes[0], output_path=output_path, show=not args.no_show)
            
            if i >= 2:  # Process only 3 samples
                break
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 