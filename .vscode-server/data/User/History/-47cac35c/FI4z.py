import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dataloader.lowres_image_dataloader import get_lowres_image_dataloader
from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader

def visualize_lowres_batch(batch, output_path=None, show=True):
    """
    Visualize a batch of lowres 3D volumes.
    
    Args:
        batch: Batch of data from the lowres dataloader
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
    """
    if 'image' in batch:
        # Raw images from lowres_image_dataloader
        volumes = batch['image']  # Shape [B, Z, 1, H, W]
    else:
        # Images already adapted for texture model
        volumes = batch[0]  # Shape [B, 1, Z, H, W]
        volumes = volumes.transpose(1, 2)  # Convert to [B, Z, 1, H, W] for visualization
    
    # Create a figure for all samples in batch
    batch_size = volumes.shape[0]
    fig = plt.figure(figsize=(16, 4 * batch_size))
    
    # For each sample in the batch
    for b in range(batch_size):
        volume = volumes[b]  # Shape [Z, 1, H, W]
        depth = volume.shape[0]
        
        # Select a subset of slices to visualize (start, middle, end)
        slice_indices = [0, depth // 4, depth // 2, 3 * depth // 4, depth - 1]
        
        # Create subplot for each slice
        for i, z_idx in enumerate(slice_indices):
            if z_idx >= depth:
                continue
                
            # Get the slice and remove channel dimension
            slice_img = volume[z_idx, 0].cpu().numpy()
            
            # Normalize for better visualization if not already done
            if slice_img.max() > 1.0:
                slice_img = slice_img / 255.0
                
            # Convert from [-1, 1] to [0, 1] range if needed
            if slice_img.min() < 0:
                slice_img = (slice_img + 1) / 2
            
            # Create subplot    
            ax = fig.add_subplot(batch_size, len(slice_indices), b * len(slice_indices) + i + 1)
            
            # Display the image
            ax.imshow(slice_img, cmap='gray')
            ax.set_title(f"Sample {b}, Slice {z_idx}")
            ax.axis('off')
    
    # Add metadata if available
    if 'metadata' in batch:
        metadata = batch['metadata']
        if 'sample_id' in metadata:
            plt.suptitle(f"Sample IDs: {metadata['sample_id']}\nClass: {metadata.get('class_name', 'N/A')}")
    
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

def visualize_lowres_3d(batch, output_path=None, show=True):
    """
    Create a 3D visualization of a lowres volume.
    
    Args:
        batch: Batch of data from the lowres dataloader
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("3D visualization requires matplotlib with 3D support.")
        return
        
    # Get the first volume in the batch
    if 'image' in batch:
        # Raw images from lowres_image_dataloader
        volume = batch['image'][0]  # Shape [Z, 1, H, W]
    else:
        # Images already adapted for texture model
        volume = batch[0][0]  # Shape [1, Z, H, W]
        volume = volume.transpose(0, 1)  # Convert to [Z, 1, H, W]
    
    # Get volume dimensions
    depth, _, height, width = volume.shape
    
    # Create coordinates for plotting
    z, y, x = np.mgrid[:depth, :height, :width]
    
    # Create thresholded volume for visualization (binary mask)
    vol_data = volume[:, 0].cpu().numpy()
    
    # Normalize for better visualization if not already done
    if vol_data.max() > 1.0:
        vol_data = vol_data / 255.0
        
    # Convert from [-1, 1] to [0, 1] range if needed
    if vol_data.min() < 0:
        vol_data = (vol_data + 1) / 2
    
    # Threshold to create binary volume for visualization
    threshold = 0.5
    vol_binary = vol_data > threshold
    
    # Create coordinates for visualization (points where volume is above threshold)
    z_coords, y_coords, x_coords = np.where(vol_binary)
    
    # Create figure for 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, alpha=0.1, s=1, c=vol_data[z_coords, y_coords, x_coords])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add metadata if available
    if 'metadata' in batch:
        metadata = batch['metadata']
        if 'sample_id' in metadata:
            ax.set_title(f"Sample ID: {metadata['sample_id'][0]}\nClass: {metadata.get('class_name', ['N/A'])[0]}")
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"3D visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_lowres_projections(batch, output_path=None, show=True):
    """
    Visualize projections of a lowres 3D volume along each axis.
    
    Args:
        batch: Batch of data from the lowres dataloader
        output_path: Path to save the visualization (optional)
        show: Whether to show the plot
    """
    # Get the first volume in the batch
    if 'image' in batch:
        # Raw images from lowres_image_dataloader
        volume = batch['image'][0]  # Shape [Z, 1, H, W]
    else:
        # Images already adapted for texture model
        volume = batch[0][0]  # Shape [1, Z, H, W]
        volume = volume.transpose(0, 1)  # Convert to [Z, 1, H, W]
    
    # Get volume dimensions
    depth, _, height, width = volume.shape
    vol_data = volume[:, 0].cpu().numpy()
    
    # Normalize for better visualization if not already done
    if vol_data.max() > 1.0:
        vol_data = vol_data / 255.0
        
    # Convert from [-1, 1] to [0, 1] range if needed
    if vol_data.min() < 0:
        vol_data = (vol_data + 1) / 2
    
    # Create max intensity projections along each axis
    proj_z = np.max(vol_data, axis=0)  # Top-down view
    proj_y = np.max(vol_data, axis=1)  # Front view
    proj_x = np.max(vol_data, axis=2)  # Side view
    
    # Create figure for projections
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot projections
    axs[0].imshow(proj_z, cmap='gray')
    axs[0].set_title('Top-down view (Z projection)')
    axs[0].axis('off')
    
    axs[1].imshow(proj_y, cmap='gray')
    axs[1].set_title('Front view (Y projection)')
    axs[1].axis('off')
    
    axs[2].imshow(proj_x, cmap='gray')
    axs[2].set_title('Side view (X projection)')
    axs[2].axis('off')
    
    # Add metadata if available
    if 'metadata' in batch:
        metadata = batch['metadata']
        if 'sample_id' in metadata:
            plt.suptitle(f"Sample ID: {metadata['sample_id'][0]}\nClass: {metadata.get('class_name', ['N/A'])[0]}")
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Projection visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize lowres texture samples")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--sample_ids", type=str, nargs="+", help="Specific sample IDs to visualize")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--is_cytoplasm", action="store_true", help="Use cytoplasm box size (144³) instead of nucleus (104³)")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots, just save them")
    parser.add_argument("--adapted", action="store_true", help="Visualize adapted data (as processed by texture model)")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataloader
    if args.adapted:
        # Get adapted dataloader for texture model
        box_size = (144, 144, 144) if args.is_cytoplasm else (104, 104, 104)
        dataloader = get_morphofeatures_texture_dataloader(
            root_dir="low_res_dataset",
            class_csv_path=args.class_csv,
            batch_size=args.batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing pickling issues
            sample_ids=args.sample_ids,
            is_cytoplasm=args.is_cytoplasm,
            box_size=box_size,
            debug=True
        )
        print(f"Loaded adapted dataloader with {len(dataloader.dataset)} samples")
    else:
        # Get raw lowres dataloader
        dataloader = get_lowres_image_dataloader(
            root_dir=args.root_dir,
            class_csv_path=args.class_csv,
            batch_size=args.batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing pickling issues
            sample_ids=args.sample_ids,
            debug=True
        )
        print(f"Loaded raw dataloader with {len(dataloader.dataset)} samples")
    
    # Process batches
    for i, batch in enumerate(dataloader):
        prefix = "adapted" if args.adapted else "raw"
        batch_name = f"{prefix}_batch_{i}"
        type_name = "cytoplasm" if args.is_cytoplasm else "nucleus"
        
        # Create visualization of slices
        output_path = os.path.join(args.output_dir, f"{batch_name}_{type_name}_slices.png") if args.output_dir else None
        visualize_lowres_batch(batch, output_path=output_path, show=not args.no_show)
        
        # Create 3D visualization
        output_path = os.path.join(args.output_dir, f"{batch_name}_{type_name}_3d.png") if args.output_dir else None
        visualize_lowres_3d(batch, output_path=output_path, show=not args.no_show)
        
        # Create projection visualization
        output_path = os.path.join(args.output_dir, f"{batch_name}_{type_name}_projections.png") if args.output_dir else None
        visualize_lowres_projections(batch, output_path=output_path, show=not args.no_show)
        
        # Only process 2 batches to avoid too many visualizations
        if i >= 1:
            break
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 