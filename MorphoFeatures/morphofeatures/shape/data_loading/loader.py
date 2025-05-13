import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Import our custom dataloader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from dataloader.mesh_dataloader import get_mesh_dataloader_v2, MeshDataset
from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader, adapt_mesh_dataloader_for_morphofeatures

def get_train_val_loaders(data_config, loader_config):
    """
    Get train and validation dataloaders.
    This is a wrapper around our custom dataloaders to match the API expected by MorphoFeatures.
    
    Args:
        data_config (dict): Configuration for the dataset
        loader_config (dict): Configuration for the dataloader
        
    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    # Get the mesh dataloader with our custom function
    dataloader = get_mesh_dataloader_v2(
        root_dir=data_config.get('root_dir', 'data'),
        class_csv_path=data_config.get('class_csv_path'),
        batch_size=loader_config.get('batch_size', 8),
        shuffle=loader_config.get('shuffle', True),
        num_workers=loader_config.get('num_workers', 4),
        precomputed_dir=data_config.get('precomputed_dir'),
        return_type='pointcloud',  # Always use pointcloud for MorphoFeatures
        num_points=data_config.get('num_points', 1024),
        cache_dir=data_config.get('cache_dir')
    )
    
    # If a validation split is provided, create train/val sets
    if data_config.get('val_split', 0) > 0:
        # Get the full dataset
        dataset = dataloader.dataset
        
        # Get indices for splitting
        indices = list(range(len(dataset)))
        split = int(np.floor(data_config['val_split'] * len(dataset)))
        
        # Shuffle indices if requested
        if loader_config.get('shuffle', True):
            seed = data_config.get('seed', 42)
            np.random.seed(seed)
            np.random.shuffle(indices)
        
        # Split indices
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create train and validation dataloaders
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=loader_config.get('batch_size', 8),
            shuffle=loader_config.get('shuffle', True),
            num_workers=loader_config.get('num_workers', 4),
            collate_fn=dataloader.collate_fn
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=loader_config.get('batch_size', 8),
            shuffle=False,  # No need to shuffle validation data
            num_workers=loader_config.get('num_workers', 4),
            collate_fn=dataloader.collate_fn
        )
    else:
        # If no validation split provided, just return the same dataloader for both
        train_loader = dataloader
        val_loader = dataloader
    
    # Adapt the dataloaders for MorphoFeatures if needed
    train_loader = adapt_mesh_dataloader_for_morphofeatures(train_loader)
    val_loader = adapt_mesh_dataloader_for_morphofeatures(val_loader)
    
    return {
        'train': train_loader,
        'val': val_loader
    }

def get_simple_loader(data_config, loader_config):
    """
    Get a simple (non-train/val split) dataloader.
    This is used by the generate_shape_embeddings.py script.
    
    Args:
        data_config (dict): Configuration for the dataset
        loader_config (dict): Configuration for the dataloader
        
    Returns:
        DataLoader: A dataloader for the dataset
    """
    # Get the mesh dataloader with our custom function
    dataloader = get_mesh_dataloader_v2(
        root_dir=data_config.get('root_dir', 'data'),
        class_csv_path=data_config.get('class_csv_path'),
        batch_size=loader_config.get('batch_size', 8),
        shuffle=False,  # Don't shuffle for embedding generation
        num_workers=loader_config.get('num_workers', 4),
        precomputed_dir=data_config.get('precomputed_dir'),
        return_type='pointcloud',  # Always use pointcloud for MorphoFeatures
        num_points=data_config.get('num_points', 1024),
        cache_dir=data_config.get('cache_dir')
    )
    
    # Adapt the dataloader for MorphoFeatures if needed
    adapted_loader = adapt_mesh_dataloader_for_morphofeatures(dataloader)
    
    return adapted_loader 