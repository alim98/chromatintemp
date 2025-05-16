import os
import torch
import numpy as np
import argparse
from tqdm import tqdm

from dataloader.mesh_dataloader import get_mesh_dataloader_v2

def reshape_for_morphofeatures(batch):
    """
    Reshape the batched data to match MorphoFeatures model expectations.
    
    Args:
        batch (dict): Batch data from the mesh dataloader
        
    Returns:
        dict: Reshaped batch for MorphoFeatures
    """
    # We noticed the points are already in the correct format [B, 3, N, 1]
    # and features is already provided in the right format [B, 6, N, 1]
    points = batch['points']
    features = batch.get('features')
    
    if features is None and 'normals' in batch:
        # If features is missing but we have normals, we'd need to create it
        normals = batch['normals']
        # However, your normals are [B, N, 3] but points are [B, 3, N, 1]
        # So we need to reshape normals to match
        if len(normals.shape) == 3:  # If normals is [B, N, 3]
            normals_reshaped = normals.transpose(1, 2).unsqueeze(-1)  # Now [B, 3, N, 1]
            # Concatenate along the feature dimension
            points_squeezed = points.squeeze(-1)  # [B, 3, N]
            normals_squeezed = normals_reshaped.squeeze(-1)  # [B, 3, N]
            features = torch.cat([points_squeezed, normals_squeezed], dim=1).unsqueeze(-1)  # [B, 6, N, 1]
    
    # Create a new batch with the correct tensors
    morpho_batch = {
        'points': points,
        'features': features if features is not None else batch.get('features', points),
        'label': batch['label'],
        'metadata': batch['metadata']
    }
    
    return morpho_batch

def test_deepgcn_model(model, batch):
    """
    Test if a batch is compatible with MorphoFeatures DeepGCN model.
    
    Args:
        model: DeepGCN model instance
        batch (dict): Batch data formatted for MorphoFeatures
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        points = batch['points']
        features = batch['features']
        
        print(f"Input shapes: points={points.shape}, features={features.shape}")
        
        # Forward pass
        out, h = model(points, features)
        
        print(f"Output shapes: out={out.shape}, h={h.shape}")
        return True
    except Exception as e:
        print(f"Error running model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test mesh dataloader compatibility with MorphoFeatures")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points per mesh")
    args = parser.parse_args()
    
    # Create a dataloader
    dataloader = get_mesh_dataloader_v2(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        batch_size=args.batch_size,
        return_type='pointcloud',
        num_points=args.num_points,
        debug=True
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Test the first batch
    try:
        for batch in dataloader:
            print("\nOriginal batch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: (dict)")
            
            # If features already exists in the batch, use it directly
            if 'features' in batch:
                morpho_batch = batch  # Already in the correct format
            else:
                # Reshape the batch for MorphoFeatures
                morpho_batch = reshape_for_morphofeatures(batch)
            
            print("\nReshaped batch for MorphoFeatures:")
            for key, value in morpho_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: (dict)")
            
            # Check DeepGCN compatibility if available
            try:
                from MorphoFeatures.morphofeatures.shape.network import DeepGCN
                
                print("\nTesting compatibility with DeepGCN model...")
                model = DeepGCN(
                    in_channels=6,  # 3D coordinates + normals
                    channels=64,
                    out_channels=64
                )
                
                is_compatible = test_deepgcn_model(model, morpho_batch)
                if is_compatible:
                    print("\n✅ Mesh dataloader is compatible with MorphoFeatures shape model!")
                else:
                    print("\n❌ Mesh dataloader is NOT compatible with MorphoFeatures shape model.")
            except ImportError:
                print("\nMorphoFeatures DeepGCN model not available for testing.")
                print("You can still check if the shapes match what's expected:")
                print("  - points should be [B, 3, N, 1]")
                print("  - features should be [B, 6, N, 1]")
            
            # Only test the first batch
            break
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 