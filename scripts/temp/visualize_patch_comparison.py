import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dataloaders
from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader
from dataloader.highres_texture_adapter import get_morphofeatures_highres_texture_dataloader, extract_cubes

def extract_highres_cubes_from_lowres(lowres_volume, num_cubes=8):
    """
    Simulate extracting highres cubes from a lowres volume.
    
    Args:
        lowres_volume (numpy.ndarray): Lowres volume with shape [Z, H, W]
        num_cubes (int): Number of cubes to extract
        
    Returns:
        list: List of cube coordinates (z_start, z_end, y_start, y_end, x_start, x_end)
    """
    depth, height, width = lowres_volume.shape
    cube_size = 32
    cube_coords = []
    
    # Generate random positions for cubes
    for _ in range(num_cubes):
        z_start = np.random.randint(0, depth - cube_size)
        y_start = np.random.randint(0, height - cube_size)
        x_start = np.random.randint(0, width - cube_size)
        
        cube_coords.append((
            z_start, z_start + cube_size,
            y_start, y_start + cube_size,
            x_start, x_start + cube_size
        ))
    
    return cube_coords

def normalize_for_display(image):
    """
    Normalize an image for display.
    
    Args:
        image (numpy.ndarray): Input image with shape [H, W] or [H, W, 3]
        
    Returns:
        numpy.ndarray: Normalized image with shape [H, W] or [H, W, 3]
    """
    if image.ndim == 2:
        # For grayscale images
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return image
    elif image.ndim == 3:
        # For color images
        min_val = image.min()
        max_val = image.max()
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        else:
            return image
    else:
        raise ValueError("Unsupported image dimension")

def visualize_patch_comparison(lowres_batch, highres_batch, output_path=None, show=True):
    """
    Visualize the relationship between lowres and highres patches.
    
    Args:
        lowres_batch: Batch from the lowres texture dataloader
        highres_batch: Batch from the highres texture dataloader
        output_path: Path to save the visualization
        show: Whether to show the plot
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract volumes
    lowres_volume = lowres_batch[0][0, 0].cpu().numpy()  # First sample, first channel [Z, H, W]
    highres_views = [
        highres_batch[0][0, 0].cpu().numpy(),  # First sample, first channel, view 1
        highres_batch[1][0, 0].cpu().numpy()   # First sample, first channel, view 2
    ]
    
    # Get dimensions
    lowres_depth, lowres_height, lowres_width = lowres_volume.shape
    highres_depth, highres_height, highres_width = highres_views[0].shape
    
    # Calculate how many highres cubes would fit in the lowres volume
    cubes_z = lowres_depth // highres_depth
    cubes_y = lowres_height // highres_height
    cubes_x = lowres_width // highres_width
    total_cubes = cubes_z * cubes_y * cubes_x
    
    # Generate random positions for cubes
    cube_coords = extract_highres_cubes_from_lowres(lowres_volume, num_cubes=8)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Show lowres volume with cube positions
    # Mid z-slice of lowres volume
    z_mid = lowres_depth // 2
    lowres_slice = lowres_volume[z_mid].copy()
    
    # Normalize for visualization - consistent method for all images
    lowres_slice = normalize_for_display(lowres_slice)
    
    # Plot lowres slice
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(lowres_slice, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f"Low-Resolution Volume\n104³ voxels at 80×80×100nm\nMid Z-slice ({z_mid}/{lowres_depth})")
    
    # Highlight random cube positions
    colors = plt.cm.tab10(np.linspace(0, 1, len(cube_coords)))
    for i, (_, _, y_start, y_end, x_start, x_end) in enumerate(cube_coords):
        rect = mpatches.Rectangle(
            (x_start, y_start), 
            x_end - x_start, 
            y_end - y_start, 
            linewidth=2, 
            edgecolor=colors[i], 
            facecolor='none',
            label=f"Cube {i+1}"
        )
        ax1.add_patch(rect)
    
    # Add legend for cube positions
    ax1.legend(loc='upper right', fontsize='small')
    
    # 2. Show a single highres cube at the same scale
    ax2 = fig.add_subplot(2, 3, 2)
    # Mid z-slice of first highres view
    highres_z_mid = highres_depth // 2
    highres_slice = highres_views[0][highres_z_mid].copy()
    
    # Use consistent normalization for visualization
    highres_slice = normalize_for_display(highres_slice)
    
    # Plot highres slice (scaled to match lowres)
    ax2.imshow(highres_slice, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"High-Resolution Cube (at same scale)\n32³ voxels at 20×20×25nm\nMid Z-slice ({highres_z_mid}/{highres_depth})")
    
    # 3. Show zoomed-in highres cube
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(highres_slice, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f"High-Resolution Cube (zoomed in)\n32³ voxels at 20×20×25nm\nMid Z-slice ({highres_z_mid}/{highres_depth})")
    
    # 4. Diagram showing cubes within volume
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Create a wireframe for the lowres volume
    lowres_corners = np.array([
        [0, 0, 0],
        [lowres_width, 0, 0],
        [lowres_width, lowres_height, 0],
        [0, lowres_height, 0],
        [0, 0, lowres_depth],
        [lowres_width, 0, lowres_depth],
        [lowres_width, lowres_height, lowres_depth],
        [0, lowres_height, lowres_depth]
    ])
    
    # Plot lowres volume edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for start, end in edges:
        ax4.plot(
            [lowres_corners[start, 0], lowres_corners[end, 0]],
            [lowres_corners[start, 1], lowres_corners[end, 1]],
            [lowres_corners[start, 2], lowres_corners[end, 2]],
            'k-', alpha=0.5
        )
    
    # Plot highres cubes within the volume
    for i, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(cube_coords):
        cube_corners = np.array([
            [x_start, y_start, z_start],
            [x_end, y_start, z_start],
            [x_end, y_end, z_start],
            [x_start, y_end, z_start],
            [x_start, y_start, z_end],
            [x_end, y_start, z_end],
            [x_end, y_end, z_end],
            [x_start, y_end, z_end]
        ])
        
        # Plot cube edges
        for start, end in edges:
            ax4.plot(
                [cube_corners[start, 0], cube_corners[end, 0]],
                [cube_corners[start, 1], cube_corners[end, 1]],
                [cube_corners[start, 2], cube_corners[end, 2]],
                color=colors[i], alpha=0.7
            )
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title("32³ Cubes Within 104³ Volume")
    
    # Set equal aspect ratio
    max_range = max(lowres_width, lowres_height, lowres_depth)
    ax4.set_xlim(0, max_range)
    ax4.set_ylim(0, max_range)
    ax4.set_zlim(0, max_range)
    
    # 5. Comparison of multiple highres cubes
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Create a grid to place multiple cubes
    grid_size = int(np.ceil(np.sqrt(min(6, len(highres_views) * 2))))
    grid_img = np.zeros((grid_size * highres_height, grid_size * highres_width))
    
    # We'll fill the grid with views from the highres batch
    count = 0
    for view_idx in range(min(3, len(highres_views))):
        for z_idx in range(2):  # Show 2 z-slices per view
            if count >= grid_size * grid_size:
                break
                
            row = count // grid_size
            col = count % grid_size
            
            # Get the slice
            z = highres_depth // 3 * (z_idx + 1)  # Show slices at 1/3 and 2/3 depth
            if z >= highres_depth:
                z = highres_depth - 1
            
            # Get and normalize the slice
            slice_img = highres_views[view_idx][z].copy()
            slice_img = normalize_for_display(slice_img)
            
            # Add to grid
            grid_img[row*highres_height:(row+1)*highres_height, 
                     col*highres_width:(col+1)*highres_width] = slice_img
            count += 1
    
    ax5.imshow(grid_img, cmap='gray', vmin=0, vmax=1)
    ax5.set_title(f"Multiple High-Resolution Cubes\nShowing {min(6, len(highres_views) * 2)} slices from different cubes")
    
    # 6. Add text description with statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    lowres_volume_size = lowres_depth * lowres_height * lowres_width
    highres_cube_size = highres_depth * highres_height * highres_width
    
    # Ratio of resolutions
    resolution_ratio_xy = 80 / 20  # 80nm vs 20nm 
    resolution_ratio_z = 100 / 25  # 100nm vs 25nm
    
    # Number of bytes per volume
    lowres_bytes = lowres_volume_size * 4  # 4 bytes per float32
    highres_bytes = highres_cube_size * 4  # 4 bytes per float32
    
    description = [
        "COMPARISON OF LOW-RES AND HIGH-RES TEXTURE INPUTS",
        "",
        f"Low-Resolution Volume:",
        f"• Size: {lowres_depth}×{lowres_height}×{lowres_width} = {lowres_volume_size:,} voxels",
        f"• Resolution: 80×80×100nm per voxel",
        f"• Memory: {lowres_bytes/1024/1024:.1f} MB per volume",
        f"• Purpose: Captures medium-scale texture features",
        f"• Processing: Full volume autoencoder with skip connections",
        "",
        f"High-Resolution Cubes:",
        f"• Size: {highres_depth}×{highres_height}×{highres_width} = {highres_cube_size:,} voxels per cube",
        f"• Resolution: 20×20×25nm per voxel ({resolution_ratio_xy}× higher in XY, {resolution_ratio_z}× in Z)",
        f"• Memory: {highres_bytes/1024:.1f} KB per cube",
        f"• Cubes Per Volume: Up to {total_cubes} cubes (theoretical maximum)",
        f"• Typical Use: {6-12} cubes per sample after filtering",
        f"• Purpose: Captures organelle-scale ultrastructure (mitochondrial cristae, ER sheets, chromatin granularity)",
        f"• Processing: Contrastive learning on cube pairs with data augmentation",
        "",
        f"In the paper, High-Res branch is crucial for discriminating:",
        f"• Mid-gut vs. secretory cells",
        f"• Rhabdomeric photoreceptors",
        f"• Chromatin-rich muscles",
        f"• Other fine ultrastructural details invisible at lower resolution"
    ]
    
    ax6.text(0, 1, '\n'.join(description), va='top', ha='left', fontsize=10, linespacing=1.5)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Add overall title
    fig.suptitle("MorphoFeatures: Low-Resolution vs. High-Resolution Texture Inputs", fontsize=16, y=0.98)
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Patch comparison visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize the relationship between lowres and highres patches")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--sample_id", type=str, help="Sample ID to visualize")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots, just save them")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get lowres batch
    lowres_dataloader = get_morphofeatures_texture_dataloader(
        root_dir="data",
        batch_size=1,
        class_csv_path="chromatin_classes_and_samples.csv",
        sample_ids=[args.sample_id] if args.sample_id else None,
        num_workers=0,
        debug=False
    )
    
    # Get highres batch
    highres_dataloader = get_morphofeatures_highres_texture_dataloader(
        root_dir="data",
        batch_size=1,
        class_csv_path="chromatin_classes_and_samples.csv",
        sample_ids=[args.sample_id] if args.sample_id else None,
        num_workers=0,
        debug=False
    )
    
    # Get a batch from each
    for lowres_batch in lowres_dataloader:
        for highres_batch in highres_dataloader:
            # Create visualization
            output_path = os.path.join(args.output_dir, f"patch_comparison_{args.sample_id or 'random'}.png") if args.output_dir else None
            visualize_patch_comparison(lowres_batch, highres_batch, output_path=output_path, show=not args.no_show)
            break
        break
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 