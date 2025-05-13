import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dataloaders
from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader
from dataloader.highres_texture_adapter import get_morphofeatures_highres_texture_dataloader, extract_cubes
from MorphoFeatures.morphofeatures.shape.data_loading.loader import get_train_val_loaders

def load_shape_model_input(sample_id=None, config=None):
    """
    Load an example input for the shape model.
    
    Args:
        sample_id (str): Sample ID to visualize (optional)
        config (dict): Configuration for the shape dataloader
        
    Returns:
        dict: Dictionary with 'points' and 'faces' tensors
    """
    # First try to load pre-generated sample mesh
    try:
        sample_mesh_path = "sample_meshes/sample_mesh.pt"
        if os.path.exists(sample_mesh_path):
            print(f"Loading sample mesh from {sample_mesh_path}")
            return torch.load(sample_mesh_path)
    except Exception as e:
        print(f"Error loading sample mesh: {e}")
    
    if config is None:
        # Default configuration for shape model
        config = {
            'data': {
                'root_dir': 'data',
                'cell_dirs': ['cells'],
                'split': {'train': 0.8, 'val': 0.2},
                'use_gt': True,
                'add_noise': False
            },
            'loader': {
                'batch_size': 1,
                'num_workers': 0,
                'shuffle': True,
                'drop_last': False
            }
        }
    
    # Try to get dataloaders for shape model
    try:
        loaders = get_train_val_loaders(config['data'], config['loader'])
        train_loader = loaders['train']
        
        # Get first batch
        for batch in train_loader:
            return batch
    except Exception as e:
        print(f"Error loading shape model input: {e}")
    
    # If all else fails, generate a simple mesh
    try:
        print("Generating simple mesh as fallback")
        from generate_sample_mesh import generate_cell_like_mesh
        return generate_cell_like_mesh(n_vertices=1000, noise_level=0.2)
    except Exception as e:
        print(f"Error generating mesh: {e}")
        
    # Last resort: return dummy data
    print("Returning dummy mesh data")
    return {
        'points': torch.randn(2, 1000, 3),  # Two point clouds
        'features': torch.randn(2, 1000, 3),  # Features for each point
        'faces': torch.tensor([[[0, 1, 2], [1, 2, 3]]], dtype=torch.int64)  # Dummy faces
    }

def visualize_shape_input(batch, ax=None, show=True):
    """
    Visualize input to the shape model (3D mesh).
    
    Args:
        batch: Batch from the shape model dataloader
        ax: Matplotlib axis to plot on
        show: Whether to show the plot
        
    Returns:
        fig: Matplotlib figure if show=True
    """
    # Extract point cloud from batch
    points = batch['points'][0].cpu().numpy()  # Take first sample from batch
    
    # Create new figure if no axis provided
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        create_new_fig = True
    else:
        create_new_fig = False
    
    # For proper mesh visualization, we need to:
    # 1. Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.3, color='blue')
    
    # 2. Attempt to visualize the mesh structure
    try:
        # If batch contains faces information
        if 'faces' in batch:
            faces = batch['faces'][0].cpu().numpy()
            # Plot mesh triangles
            for face in faces:
                # Get vertices for this face
                verts = points[face]
                # Plot the triangle
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                                alpha=0.5, color='cyan', shade=True)
        else:
            # If no explicit face information, attempt to visualize local connectivity
            from scipy.spatial import Delaunay
            # Use a subset of points to make this computationally feasible
            if len(points) > 500:
                # Subsample points
                indices = np.random.choice(len(points), 500, replace=False)
                subset_points = points[indices]
            else:
                subset_points = points
                
            # Create a simplified triangulation for visualization (not the actual mesh)
            try:
                # Attempt 3D triangulation (may fail for complex shapes)
                tri = Delaunay(subset_points)
                # Plot a small subset of the triangles to avoid overcrowding
                max_faces = 200
                if len(tri.simplices) > max_faces:
                    face_indices = np.random.choice(len(tri.simplices), max_faces, replace=False)
                    simplices = tri.simplices[face_indices]
                else:
                    simplices = tri.simplices
                
                for simplex in simplices:
                    verts = subset_points[simplex]
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                                    alpha=0.2, color='cyan', shade=True)
            except Exception as e:
                # Fall back to wireframe visualization
                print(f"Triangulation failed: {e}, falling back to wireframe")
                # Calculate nearest neighbors for wireframe
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=4).fit(subset_points)
                distances, indices = nn.kneighbors(subset_points)
                
                # Plot edges between neighbors
                for i, nbrs in enumerate(indices):
                    for j in nbrs[1:]:  # Skip the point itself (first neighbor)
                        # Plot edge
                        ax.plot([subset_points[i, 0], subset_points[j, 0]],
                                [subset_points[i, 1], subset_points[j, 1]],
                                [subset_points[i, 2], subset_points[j, 2]],
                                color='cyan', alpha=0.1)
    except Exception as e:
        print(f"Error visualizing mesh: {e}, showing points only")
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Shape Model Input (3D Mesh)')
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(points[:, 0]),
        np.ptp(points[:, 1]),
        np.ptp(points[:, 2])
    ])
    mid_x = np.mean([points[:, 0].min(), points[:, 0].max()])
    mid_y = np.mean([points[:, 1].min(), points[:, 1].max()])
    mid_z = np.mean([points[:, 2].min(), points[:, 2].max()])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    if create_new_fig and show:
        plt.tight_layout()
        plt.show()
        return fig
    
    return None

def visualize_lowres_input(batch, ax=None, show=True):
    """
    Visualize input to the lowres texture model.
    
    Args:
        batch: Batch from the lowres texture dataloader
        ax: Matplotlib axis to plot on
        show: Whether to show the plot
        
    Returns:
        fig: Matplotlib figure if show=True
    """
    # Extract volume from batch
    volume = batch[0][0, 0].cpu().numpy()  # Take first sample, first channel
    
    # Create new figure if no axis provided
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        create_new_fig = True
    else:
        axes = [ax]
        create_new_fig = False
    
    # Get volume dimensions
    depth, height, width = volume.shape
    
    # Choose three representative slices
    z_indices = [depth//4, depth//2, 3*depth//4]
    
    # Plot each slice
    for i, z_idx in enumerate(z_indices):
        if i >= len(axes):
            break
            
        # Get the slice
        slice_img = volume[z_idx]
        
        # Normalize for visualization if needed
        if slice_img.max() > 1.0:
            slice_img = slice_img / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice_img.min() < 0:
            slice_img = (slice_img + 1) / 2
        
        # Plot the slice
        im = axes[i].imshow(slice_img, cmap='gray')
        axes[i].set_title(f"Z-slice {z_idx}/{depth}")
        axes[i].axis('off')
    
    if create_new_fig:
        plt.suptitle("Low-Resolution Texture Model Input (104³ volume)", fontsize=14)
        plt.tight_layout()
        if show:
            plt.show()
            return fig
    
    return None

def visualize_highres_input(batch, ax=None, show=True):
    """
    Visualize input to the highres texture model.
    
    Args:
        batch: Batch from the highres texture dataloader
        ax: Matplotlib axis to plot on
        show: Whether to show the plot
        
    Returns:
        fig: Matplotlib figure if show=True
    """
    # Extract a contrastive pair from batch
    view1 = batch[0][0, 0].cpu().numpy()  # First sample, first channel, view 1
    view2 = batch[1][0, 0].cpu().numpy()  # First sample, first channel, view 2
    
    # Create new figure if no axis provided
    if ax is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        create_new_fig = True
    else:
        # If only one axis provided, we can't show all slices
        axes = [[ax]]
        create_new_fig = False
    
    # Get cube dimensions
    depth, height, width = view1.shape
    
    # Choose three representative slices
    z_indices = [depth//4, depth//2, 3*depth//4]
    
    # Plot slices for view 1
    for i, z_idx in enumerate(z_indices):
        if i >= len(axes[0]):
            break
            
        # Get the slice
        slice_img = view1[z_idx]
        
        # Normalize for visualization if needed
        if slice_img.max() > 1.0:
            slice_img = slice_img / 255.0
            
        # Convert from [-1, 1] to [0, 1] range if needed
        if slice_img.min() < 0:
            slice_img = (slice_img + 1) / 2
        
        # Plot the slice
        im = axes[0][i].imshow(slice_img, cmap='gray')
        axes[0][i].set_title(f"View 1, Z-slice {z_idx}/{depth}")
        axes[0][i].axis('off')
    
    # Plot slices for view 2 (if we have a second row of axes)
    if len(axes) > 1:
        for i, z_idx in enumerate(z_indices):
            if i >= len(axes[1]):
                break
                
            # Get the slice
            slice_img = view2[z_idx]
            
            # Normalize for visualization if needed
            if slice_img.max() > 1.0:
                slice_img = slice_img / 255.0
                
            # Convert from [-1, 1] to [0, 1] range if needed
            if slice_img.min() < 0:
                slice_img = (slice_img + 1) / 2
            
            # Plot the slice
            im = axes[1][i].imshow(slice_img, cmap='gray')
            axes[1][i].set_title(f"View 2, Z-slice {z_idx}/{depth}")
            axes[1][i].axis('off')
    
    if create_new_fig:
        plt.suptitle("High-Resolution Texture Model Input (32³ cubes for contrastive learning)", fontsize=14)
        plt.tight_layout()
        if show:
            plt.show()
            return fig
    
    return None

def visualize_comparison(sample_id=None, output_path=None, show=True):
    """
    Create a comparison visualization of all three MorphoFeatures model inputs.
    
    Args:
        sample_id (str): Sample ID to visualize (optional)
        output_path (str): Path to save visualization (optional)
        show (bool): Whether to show the visualization
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure with subplots for each model
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Shape model input (point cloud)
    ax1 = fig.add_subplot(3, 1, 1, projection='3d')
    shape_batch = load_shape_model_input(sample_id)
    visualize_shape_input(shape_batch, ax=ax1, show=False)
    
    # 2. Low-resolution texture model input
    try:
        # Get a batch from the lowres texture dataloader
        lowres_dataloader = get_morphofeatures_texture_dataloader(
            root_dir="data",
            batch_size=1,
            class_csv_path="chromatin_classes_and_samples.csv",
            sample_ids=[sample_id] if sample_id else None,
            num_workers=0,
            debug=False
        )
        
        for lowres_batch in lowres_dataloader:
            break
        
        ax2 = fig.add_subplot(3, 1, 2)
        visualize_lowres_input(lowres_batch, ax=ax2, show=False)
    except Exception as e:
        print(f"Error loading lowres texture input: {e}")
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.text(0.5, 0.5, "Lowres texture data not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax2.axis('off')
    
    # 3. High-resolution texture model input
    try:
        # Get a batch from the highres texture dataloader
        highres_dataloader = get_morphofeatures_highres_texture_dataloader(
            root_dir="data",
            batch_size=1,
            class_csv_path="chromatin_classes_and_samples.csv",
            sample_ids=[sample_id] if sample_id else None,
            num_workers=0,
            debug=False
        )
        
        for highres_batch in highres_dataloader:
            break
        
        ax3 = fig.add_subplot(3, 1, 3)
        visualize_highres_input(highres_batch, ax=ax3, show=False)
    except Exception as e:
        print(f"Error loading highres texture input: {e}")
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.text(0.5, 0.5, "Highres texture data not available", 
                 horizontalalignment='center', verticalalignment='center')
        ax3.axis('off')
    
    # Add overall title
    plt.suptitle(f"MorphoFeatures Model Inputs Comparison (Sample ID: {sample_id or 'random'})", fontsize=16)
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Comparison visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
        
    return fig

def visualize_all_three_separately(sample_id=None, output_dir="visualizations", show=True):
    """
    Create separate visualizations for all three MorphoFeatures model inputs.
    
    Args:
        sample_id (str): Sample ID to visualize (optional)
        output_dir (str): Directory to save visualizations
        show (bool): Whether to show the visualizations
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Shape model input (point cloud)
    shape_batch = load_shape_model_input(sample_id)
    shape_fig = visualize_shape_input(shape_batch, show=show)
    if shape_fig and output_dir:
        output_path = os.path.join(output_dir, f"shape_input_{sample_id or 'random'}.png")
        shape_fig.savefig(output_path)
        print(f"Shape model input visualization saved to {output_path}")
    
    # 2. Low-resolution texture model input
    try:
        # Get a batch from the lowres texture dataloader
        lowres_dataloader = get_morphofeatures_texture_dataloader(
            root_dir="data",
            batch_size=1,
            class_csv_path="chromatin_classes_and_samples.csv",
            sample_ids=[sample_id] if sample_id else None,
            num_workers=0,
            debug=False
        )
        
        for lowres_batch in lowres_dataloader:
            lowres_fig = visualize_lowres_input(lowres_batch, show=show)
            if lowres_fig and output_dir:
                output_path = os.path.join(output_dir, f"lowres_input_{sample_id or 'random'}.png")
                lowres_fig.savefig(output_path)
                print(f"Lowres texture model input visualization saved to {output_path}")
            break
    except Exception as e:
        print(f"Error loading lowres texture input: {e}")
    
    # 3. High-resolution texture model input
    try:
        # Get a batch from the highres texture dataloader
        highres_dataloader = get_morphofeatures_highres_texture_dataloader(
            root_dir="data",
            batch_size=1,
            class_csv_path="chromatin_classes_and_samples.csv",
            sample_ids=[sample_id] if sample_id else None,
            num_workers=0,
            debug=False
        )
        
        for highres_batch in highres_dataloader:
            highres_fig = visualize_highres_input(highres_batch, show=show)
            if highres_fig and output_dir:
                output_path = os.path.join(output_dir, f"highres_input_{sample_id or 'random'}.png")
                highres_fig.savefig(output_path)
                print(f"Highres texture model input visualization saved to {output_path}")
            break
    except Exception as e:
        print(f"Error loading highres texture input: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare and visualize all three MorphoFeatures model inputs")
    parser.add_argument("--sample_id", type=str, help="Sample ID to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--combined", action="store_true", help="Create combined visualization")
    parser.add_argument("--separate", action="store_true", help="Create separate visualizations")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots, just save them")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Default to both combined and separate if neither is specified
    if not args.combined and not args.separate:
        args.combined = True
        args.separate = True
    
    # Create combined visualization
    if args.combined:
        output_path = os.path.join(args.output_dir, f"morphofeatures_comparison_{args.sample_id or 'random'}.png") if args.output_dir else None
        visualize_comparison(sample_id=args.sample_id, output_path=output_path, show=not args.no_show)
    
    # Create separate visualizations
    if args.separate:
        visualize_all_three_separately(sample_id=args.sample_id, output_dir=args.output_dir, show=not args.no_show)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 