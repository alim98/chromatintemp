import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader.mesh_dataloader import get_mesh_dataloader_v2


# Define the adapted collate function outside the main function to make it pickle-friendly
def morphofeatures_adapted_collate_fn(original_collate_fn, batch_list):
    # Call original collate function
    original_batch = original_collate_fn(batch_list)
    # Adapt the batch
    return adapt_batch_for_morphofeatures(original_batch)


def adapt_batch_for_morphofeatures(batch):
    """
    Adapt a single batch for MorphoFeatures shape model.
    
    Args:
        batch: A batch dictionary from the dataloader
        
    Returns:
        Adapted batch dictionary
    """
    # Extract and reshape tensors if needed
    morpho_batch = {}
    
    # Copy all non-tensor values directly
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            morpho_batch[key] = value
    
    # Process points
    if 'points' in batch:
        points = batch['points']
        # For DeepGCN model, points should be in shape [B, 3, N]
        if len(points.shape) == 3 and points.shape[2] == 3:  # If [B, N, 3]
            morpho_batch['points'] = points.transpose(1, 2)  # [B, 3, N]
        elif len(points.shape) == 4 and points.shape[3] == 1:  # If [B, 3, N, 1]
            morpho_batch['points'] = points.squeeze(-1)  # [B, 3, N]
        else:
            morpho_batch['points'] = points
    
    # Process features if we need to create them from points and normals
    if 'features' not in batch and 'normals' in batch and 'points' in batch:
        points = batch['points']
        normals = batch['normals']
        
        # Reshape points if needed
        if len(points.shape) == 3 and points.shape[2] == 3:  # If [B, N, 3]
            points_reshaped = points.transpose(1, 2)  # [B, 3, N]
        elif len(points.shape) == 4 and points.shape[3] == 1:  # If [B, 3, N, 1]
            points_reshaped = points.squeeze(-1)  # [B, 3, N]
        else:
            points_reshaped = points
            
        # Reshape normals if needed
        if len(normals.shape) == 3 and normals.shape[2] == 3:  # If [B, N, 3]
            normals_reshaped = normals.transpose(1, 2)  # [B, 3, N]
        elif len(normals.shape) == 4 and normals.shape[3] == 1:  # If [B, 3, N, 1]
            normals_reshaped = normals.squeeze(-1)  # [B, 3, N]
        else:
            normals_reshaped = normals
        
        # Create features by concatenating points and normals (both should be [B, 3, N])
        features = torch.cat([points_reshaped, normals_reshaped], dim=1)  # [B, 6, N]
        morpho_batch['features'] = features
    elif 'features' in batch:
        features = batch['features']
        # Ensure features are in shape [B, C, N]
        if len(features.shape) == 4 and features.shape[3] == 1:  # If [B, C, N, 1]
            morpho_batch['features'] = features.squeeze(-1)  # [B, C, N]
        else:
            morpho_batch['features'] = features
    
    return morpho_batch


def adapt_mesh_dataloader_for_morphofeatures(dataloader_or_batch):
    """
    Adapt your mesh dataloader or batch for MorphoFeatures shape model.
    
    Args:
        dataloader_or_batch: Either a DataLoader instance or a batch dict
        
    Returns:
        If DataLoader: Returns a new DataLoader with adapted collate_fn
        If batch: Returns an adapted batch
    """
    if isinstance(dataloader_or_batch, DataLoader):
        # Create a new dataloader with adapted collate function
        original_collate_fn = dataloader_or_batch.collate_fn
        
        # Use a functools.partial to create a new function with the original collate_fn bound
        from functools import partial
        adapted_fn = partial(morphofeatures_adapted_collate_fn, original_collate_fn)
        
        # Clone the dataloader with new collate function
        new_dataloader = DataLoader(
            dataset=dataloader_or_batch.dataset,
            batch_size=dataloader_or_batch.batch_size,
            shuffle=isinstance(dataloader_or_batch.sampler, torch.utils.data.sampler.RandomSampler),
            num_workers=dataloader_or_batch.num_workers,
            collate_fn=adapted_fn,
            pin_memory=dataloader_or_batch.pin_memory,
        )
        return new_dataloader
    
    elif isinstance(dataloader_or_batch, dict):
        # Process a single batch
        return adapt_batch_for_morphofeatures(dataloader_or_batch)
    
    else:
        raise TypeError("Input must be either a DataLoader or a batch dict")


def get_morphofeatures_mesh_dataloader(
    root_dir,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    class_csv_path=None,
    sample_ids=None,
    filter_by_class=None,
    ignore_unclassified=True,
    precomputed_dir=None,
    generate_on_load=True,
    num_points=1024,
    sample_percent=100,
    cache_dir=None,
    pin_memory=False,
    debug=False
):
    """
    Get a mesh dataloader already adapted for MorphoFeatures shape model.
    
    This is a convenience function that creates a mesh dataloader and adapts it
    for the MorphoFeatures shape model.
    
    Args:
        See get_mesh_dataloader_v2 for parameter documentation
        
    Returns:
        DataLoader: Adapted dataloader for MorphoFeatures
    """
    # Create the original dataloader
    dataloader = get_mesh_dataloader_v2(
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        class_csv_path=class_csv_path,
        sample_ids=sample_ids,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        precomputed_dir=precomputed_dir,
        generate_on_load=generate_on_load,
        return_type='pointcloud',  # Always use pointcloud for MorphoFeatures
        num_points=num_points,
        sample_percent=sample_percent,
        cache_dir=cache_dir,
        pin_memory=pin_memory,
        debug=debug
    )
    
    # Adapt the dataloader for MorphoFeatures
    return adapt_mesh_dataloader_for_morphofeatures(dataloader)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MorphoFeatures adapter")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points per mesh")
    args = parser.parse_args()
    
    # Get adapted dataloader
    dataloader = get_morphofeatures_mesh_dataloader(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        batch_size=args.batch_size,
        num_points=args.num_points,
        debug=True
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Test the first batch
    for batch in dataloader:
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: (dict)")
        
        # Verify the shapes are correct for MorphoFeatures
        points_shape = batch['points'].shape
        features_shape = batch['features'].shape
        
        print(f"\nVerifying shapes for MorphoFeatures:")
        print(f"  points shape: {points_shape}")
        print(f"  features shape: {features_shape}")
        
        if len(points_shape) == 4 and points_shape[1] == 3 and len(features_shape) == 4 and features_shape[1] == 6:
            print("\n✅ Mesh dataloader is compatible with MorphoFeatures shape model!")
        else:
            print("\n❌ Mesh dataloader is NOT compatible with MorphoFeatures shape model.")
        
        # Only test the first batch
        break 