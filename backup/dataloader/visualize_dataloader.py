import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from mesh_dataloader import get_mesh_dataloader_v2

def visualize_batch(batch, save_dir=None):
    """
    Visualize a batch of 3D point clouds
    
    Args:
        batch (dict): Batch data from the dataloader
        save_dir (str, optional): Directory to save visualizations
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have the reshaped data for MorphoFeatures
    has_morpho_format = 'features' in batch and batch['features'].dim() == 4
    
    # Get sample IDs for filenames
    sample_ids = batch['metadata']['sample_id']
    
    # Visualize each sample in the batch
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        
        # Create a figure with two subplots if we have both formats
        fig = plt.figure(figsize=(15, 7 if has_morpho_format else 5))
        
        # Original format visualization
        ax1 = fig.add_subplot(1, 2 if has_morpho_format else 1, 1, projection='3d')
        if 'points' in batch:
            if batch['points'].dim() == 4:  # MorphoFeatures format [B, 3, N, 1]
                points = batch['points'][i].permute(1, 0, 2).squeeze(-1).cpu().numpy()
            else:  # Original format [B, N, 3]
                points = batch['points'][i].cpu().numpy()
                
            # Get point masks if available to show only valid points
            if 'point_masks' in batch:
                mask = batch['point_masks'][i].bool().cpu().numpy()
                points = points[mask]
            
            # Plot the points
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, alpha=0.5)
            ax1.set_title(f"Sample {sample_id} - Point Cloud")
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Make equal aspect ratio
            max_range = np.array([
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            
            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        else:
            ax1.set_title("No point cloud data available")
            
        # MorphoFeatures format visualization
        if has_morpho_format:
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            
            # Extract the 3D points from the features tensor [B, 6, N, 1]
            features = batch['features'][i].squeeze(-1).cpu().numpy()
            morpho_points = features[:3].transpose(1, 0)  # Convert to [N, 3]
            
            # Plot the points
            ax2.scatter(morpho_points[:, 0], morpho_points[:, 1], morpho_points[:, 2], c='r', s=1, alpha=0.5)
            ax2.set_title(f"Sample {sample_id} - MorphoFeatures Format")
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # Make equal aspect ratio
            max_range = np.array([
                morpho_points[:, 0].max() - morpho_points[:, 0].min(),
                morpho_points[:, 1].max() - morpho_points[:, 1].min(),
                morpho_points[:, 2].max() - morpho_points[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (morpho_points[:, 0].max() + morpho_points[:, 0].min()) * 0.5
            mid_y = (morpho_points[:, 1].max() + morpho_points[:, 1].min()) * 0.5
            mid_z = (morpho_points[:, 2].max() + morpho_points[:, 2].min()) * 0.5
            
            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{sample_id}_visualization.png"))
            plt.close()
        else:
            plt.show()

def print_batch_info(batch):
    """Print information about the batch structure and tensor shapes"""
    print("\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Visualize data from the mesh dataloader")
    parser.add_argument("--root_dir", type=str, default="data", 
                        help="Root directory containing the samples")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv",
                        help="Path to CSV file with class information")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with pre-processed meshes")
    parser.add_argument("--sample_id", type=int, default=None,
                        help="Specific sample ID to process")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_points", type=int, default=1024,
                        help="Number of points to sample from each mesh")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--show", action="store_true",
                        help="Show visualizations instead of saving them")
    
    args = parser.parse_args()
    
    # Prepare sample IDs if specified
    sample_ids = [args.sample_id] if args.sample_id is not None else None
    
    print(f"Loading data from {args.root_dir}")
    
    # Create dataloader
    loader = get_mesh_dataloader_v2(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        precomputed_dir=args.precomputed_dir,
        sample_ids=sample_ids,
        batch_size=args.batch_size,
        return_type='pointcloud',  # We need pointcloud data for visualization
        num_points=args.num_points,
        cache_dir="data/mesh_cache",
        num_workers=0,
        debug=True
    )
    
    print(f"Dataset contains {len(loader.dataset)} samples")
    
    save_dir = "/Users/ali/Documents/codes/Chromatin/results/test"
    
    # Process batches
    for batch_idx, batch in enumerate(loader):
        print(f"\nProcessing batch {batch_idx + 1}:")
        print_batch_info(batch)
        
        # Visualize the batch
        visualize_batch(batch, save_dir)
        
        if batch_idx >= 2:  # Limit to processing just a few batches
            break

if __name__ == "__main__":
    main() 