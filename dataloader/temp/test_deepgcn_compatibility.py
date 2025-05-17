import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add the parent directory to the path to import mesh_dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataloader
from mesh_dataloader import get_mesh_dataloader_v2

def create_mock_deepgcn_model(in_channels=6, channels=64, out_channels=64):
    """
    Create a mock DeepGCN model with the same input interface as the real one
    for testing data compatibility without requiring the actual model.
    """
    class MockDeepGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, channels, kernel_size=1)
            self.pool = torch.nn.AdaptiveMaxPool2d(1)
            self.fc = torch.nn.Linear(channels, out_channels)
            
        def forward(self, points, features):
            """
            Args:
                points: Point coordinates of shape [B, 3, N, 1]
                features: Point features of shape [B, 6, N, 1]
            """
            # Print shapes for verification
            print(f"Points shape: {points.shape}")
            print(f"Features shape: {features.shape}")
            
            # Simple forward pass to verify tensor dimensions
            x = self.conv1(features)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, x  # Return two tensors like the real model
    
    return MockDeepGCN()

def test_with_model(dataloader):
    """
    Test if the dataloader outputs can be correctly processed by the DeepGCN model.
    
    Args:
        dataloader: PyTorch dataloader that provides batches of mesh data
    """
    # Create mock model
    model = create_mock_deepgcn_model()
    model.eval()
    
    # Process a single batch
    with torch.no_grad():
        for batch in dataloader:
            if 'points' not in batch or 'features' not in batch:
                print("Error: Batch does not contain 'points' and 'features' keys required by DeepGCN")
                print(f"Available keys: {list(batch.keys())}")
                return False
            
            try:
                # Try to pass data through the model
                out, h = model(batch['points'], batch['features'])
                print("Success! Data format is compatible with DeepGCN")
                print(f"Model output shapes: {out.shape}, {h.shape}")
                return True
            except Exception as e:
                print(f"Error: Failed to process batch with mock DeepGCN model: {e}")
                return False

def visualize_processed_data(batch):
    """
    Visualize the data in the MorphoFeatures format.
    
    Args:
        batch (dict): Batch data from the dataloader
    """
    # Get sample IDs for visualization
    sample_ids = batch['metadata']['sample_id']
    
    # Check if we have the required data
    if 'points' not in batch or 'features' not in batch:
        print("Error: Batch does not contain both 'points' and 'features' keys")
        return
    
    # Visualize first few samples
    for i in range(min(3, len(sample_ids))):
        sample_id = sample_ids[i]
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # Plot points (3D coordinates)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        points = batch['points'][i].permute(1, 0, 2).squeeze().cpu().numpy()  # Convert to [N, 3]
        
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1, alpha=0.5)
        ax1.set_title(f"Sample {sample_id} - Points")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot features (using first 3 dimensions of features for visualization)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        features = batch['features'][i].squeeze().cpu().numpy()  # Get all features
        
        # Use first 3 dimensions of features for 3D visualization
        feature_coords = features[:3].transpose(1, 0)  # Convert to [N, 3]
        
        # Calculate color based on remaining feature dimensions (e.g., normals)
        if features.shape[0] > 3:
            normals = features[3:].transpose(1, 0)  # Get normals or other features
            colors = np.abs(normals)  # Use normals as RGB values
        else:
            colors = 'r'
            
        ax2.scatter(feature_coords[:, 0], feature_coords[:, 1], feature_coords[:, 2], 
                   c=colors if isinstance(colors, str) else colors[:, 0], 
                   s=1, alpha=0.5)
        ax2.set_title(f"Sample {sample_id} - Features")
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test compatibility with DeepGCN model")
    parser.add_argument("--root_dir", type=str, default="data", 
                        help="Root directory containing the samples")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv",
                        help="Path to CSV file with class information")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with pre-processed meshes")
    parser.add_argument("--sample_id", type=int, default=None,
                        help="Specific sample ID to process")
    parser.add_argument("--num_points", type=int, default=1024,
                        help="Number of points to sample from each mesh")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the processed data")
    
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
        return_type='pointcloud',  # We need pointcloud data for the model
        num_points=args.num_points,
        cache_dir="data/mesh_cache",
        num_workers=0,
        debug=True
    )
    
    print(f"Dataset contains {len(loader.dataset)} samples")
    
    # Process a batch
    for batch in loader:
        # Display batch information
        print("\nBatch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
        
        # Test compatibility with DeepGCN model
        print("\nTesting compatibility with DeepGCN model...")
        is_compatible = test_with_model(loader)
        
        if is_compatible and args.visualize:
            print("\nVisualizing processed data...")
            visualize_processed_data(batch)
        
        break  # Only process one batch

if __name__ == "__main__":
    main() 