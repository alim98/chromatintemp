import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm
import math
from MorphoFeatures.morphofeatures.shape.augmentations.simple_transforms import (
    RandomCompose,
    SymmetryTransform,
    AnisotropicScaleTransform,
    AxisRotationTransform
)
from MorphoFeatures.morphofeatures.shape.augmentations.arap import arap_warp
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.mesh_utils import create_mesh_from_mask, sample_points_from_mesh
from utils.pointcloud import load_mask_volume

class MeshDataset(Dataset):
    """
    Dataset for loading and processing 3D meshes from mask volumes.
    
    This dataset can:
    1. Generate and cache meshes on-the-fly
    2. Load pre-processed meshes from disk
    3. Generate point clouds sampled from mesh surfaces
    """
    def __init__(self, 
                 root_dir,
                 class_csv_path=None,
                 sample_ids=None,
                 filter_by_class=None,
                 ignore_unclassified=False,
                 precomputed_dir=None,
                 generate_on_load=True,
                 return_type='mesh',  # 'mesh', 'pointcloud', or 'both'
                 voxel_size=(1.0, 1.0, 1.0),
                 
                 smooth_iterations=10,
                 decimate_target=5000,
                 num_points=1024,
                 sample_percent=100,
                 cache_dir=None,
                 debug=False):
        """
        Args:
            root_dir (str): Root directory containing the samples.
            class_csv_path (str, optional): Path to CSV file with class information.
            sample_ids (list, optional): List of sample IDs to include.
            filter_by_class (int or list, optional): Class ID(s) to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            precomputed_dir (str, optional): Directory with pre-processed meshes.
            generate_on_load (bool): Whether to generate meshes on load if not cached/precomputed.
            return_type (str): Type of data to return - 'mesh', 'pointcloud', or 'both'.
            voxel_size (tuple): Size of each voxel (dz, dy, dx) for mesh generation.
            smooth_iterations (int): Number of Taubin smoothing iterations.
            decimate_target (int): Target number of faces after decimation.
            num_points (int): Number of points to sample from mesh surfaces.
            sample_percent (int): Percentage of samples to load per class (1-100).
            cache_dir (str, optional): Directory to cache processed meshes.
            debug (bool): Whether to print debug information.
        """
        self.root_dir = root_dir
        self.return_type = return_type
        self.voxel_size = voxel_size
        self.smooth_iterations = smooth_iterations
        self.decimate_target = decimate_target
        self.num_points = num_points
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)
        self.cache_dir = cache_dir
        self.precomputed_dir = precomputed_dir
        self.generate_on_load = generate_on_load
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "meshes"), exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "pointclouds"), exist_ok=True)
        
        # Initialize empty sample list
        self.samples = []
        
        # Load class information from CSV if provided
        if class_csv_path and os.path.exists(class_csv_path):
            try:
                # Read the CSV file
                df = pd.read_csv(class_csv_path)
                
                # Filter out unclassified samples if requested
                if ignore_unclassified:
                    df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
                
                # Filter by class if specified
                if filter_by_class is not None:
                    if isinstance(filter_by_class, int):
                        filter_by_class = [filter_by_class]
                    df = df[df['class_id'].isin(filter_by_class)]
                
                # Filter by sample ID if specified
                if sample_ids is not None:
                    sample_ids = [str(sid).zfill(4) for sid in sample_ids]
                    df = df[df['sample_id'].astype(str).apply(lambda x: x.zfill(4)).isin(sample_ids)]
                
                # Sample a percentage of each class if requested
                if self.sample_percent < 100:
                    sampled_df = pd.DataFrame()
                    for class_id, group in df.groupby('class_id'):
                        num_to_keep = max(1, int(len(group) * self.sample_percent / 100.0))
                        sampled_group = group.sample(n=num_to_keep, random_state=42)
                        sampled_df = pd.concat([sampled_df, sampled_group])
                        if self.debug:
                            print(f"Class {class_id}: Sampled {num_to_keep}/{len(group)} samples ({self.sample_percent}%)")
                    df = sampled_df
                
                # Create sample entries for each valid sample
                for _, row in df.iterrows():
                    sample_id = str(row['sample_id']).zfill(4)
                    class_id = int(row['class_id'])
                    class_name = row['class_name']
                    
                    # Skip unclassified samples if requested
                    if ignore_unclassified and (class_name == 'Unclassified' or class_id == 19):
                        continue
                    
                    # Find the sample directory
                    sample_dir = os.path.join(root_dir, sample_id)
                    
                    # Check if mask directory exists or if we have precomputed data
                    mask_dir = os.path.join(sample_dir, 'mask')
                    
                    # Check for precomputed mesh
                    precomputed_mesh = None
                    precomputed_pc = None
                    
                    if self.precomputed_dir:
                        # Format ID with 6 digits for precomputed files (MorphoFeatures format)
                        precomputed_id = sample_id.zfill(6)
                        
                        # Check for precomputed mesh
                        mesh_path = os.path.join(self.precomputed_dir, "meshes", f"{precomputed_id}_mesh.ply")
                        if os.path.exists(mesh_path):
                            precomputed_mesh = mesh_path
                        
                        # Check for precomputed point cloud
                        pc_path = os.path.join(self.precomputed_dir, "pointclouds", f"{precomputed_id}_pc.npy")
                        if os.path.exists(pc_path):
                            precomputed_pc = pc_path
                    
                    # If we need raw data and no precomputed mesh, verify mask directory
                    if not precomputed_mesh and not precomputed_pc and not os.path.exists(mask_dir):
                        if self.debug:
                            print(f"Warning: Mask directory not found for sample {sample_id}")
                        continue
                    
                    # Add sample to the list
                    self.samples.append({
                        'sample_id': sample_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'mask_dir': mask_dir,
                        'precomputed_mesh': precomputed_mesh,
                        'precomputed_pc': precomputed_pc
                    })
                
                print(f"Total number of samples to process: {len(self.samples)}")
                
            except Exception as e:
                print(f"Error loading class CSV: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Class CSV file not provided or not found: {class_csv_path}")
        
        self.apply_aug = True  # flag for point cloud augmentations
        # Build a RandomCompose augmentation similar to paper
        self.pc_augment = RandomCompose(
            SymmetryTransform(),
            AnisotropicScaleTransform(0.9, 1.1),
            AxisRotationTransform(180, 180, 180),
            center,
            normalize,
            num_compositions=2
        )
    
    def __len__(self):
        return len(self.samples)
    
    def _load_or_generate_mesh(self, sample):
        """
        Load a precomputed mesh or generate it from the mask volume.
        
        Args:
            sample (dict): Sample information
            
        Returns:
            trimesh.Trimesh: Mesh object
        """
        sample_id = sample['sample_id']
        mask_dir = sample['mask_dir']
        precomputed_mesh = sample.get('precomputed_mesh')
        
        # Check if precomputed mesh exists
        if precomputed_mesh and os.path.exists(precomputed_mesh):
            try:
                if self.debug:
                    print(f"Loading precomputed mesh for sample {sample_id}: {precomputed_mesh}")
                import trimesh
                return trimesh.load(precomputed_mesh)
            except Exception as e:
                print(f"Error loading precomputed mesh: {e}")
                # If loading fails, fall through to generation
        
        # Check if cached mesh exists
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "meshes", f"{sample_id}_mesh.ply")
            if os.path.exists(cache_file):
                try:
                    if self.debug:
                        print(f"Loading cached mesh for sample {sample_id}")
                    import trimesh
                    return trimesh.load(cache_file)
                except Exception as e:
                    print(f"Error loading cached mesh: {e}")
                    # If loading fails, fall through to generation
        
        # Generate mesh if we should generate on load and have a mask directory
        if self.generate_on_load and os.path.exists(mask_dir):
            try:
                # Load the mask volume
                mask_volume = load_mask_volume(mask_dir)
                
                # Create mesh
                mesh = create_mesh_from_mask(
                    mask_volume, 
                    threshold=0.5, 
                    voxel_size=self.voxel_size,
                    smooth_iterations=self.smooth_iterations,
                    decimate_target=self.decimate_target
                )
                
                # Cache the mesh
                if cache_file:
                    mesh.export(cache_file)
                    if self.debug:
                        print(f"Cached mesh for sample {sample_id}")
                
                return mesh
            
            except Exception as e:
                print(f"Error generating mesh for sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # If we get here, we couldn't load or generate a mesh
        print(f"Could not load or generate mesh for sample {sample_id}")
        return None
    
    def _load_or_generate_pointcloud(self, sample, mesh=None):
        """
        Load a precomputed point cloud or generate it from the mesh.
        
        Args:
            sample (dict): Sample information
            mesh (trimesh.Trimesh, optional): Mesh to sample from if not precomputed
            
        Returns:
            torch.Tensor: Point cloud tensor of shape (N, 3) or (N, 6) with normals
        """
        sample_id = sample['sample_id']
        precomputed_pc = sample.get('precomputed_pc')
        
        # Check if precomputed point cloud exists
        if precomputed_pc and os.path.exists(precomputed_pc):
            try:
                if self.debug:
                    print(f"Loading precomputed point cloud for sample {sample_id}: {precomputed_pc}")
                pc = np.load(precomputed_pc)
                return torch.tensor(pc, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading precomputed point cloud: {e}")
                # If loading fails, fall through to generation
        
        # Check if cached point cloud exists
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "pointclouds", f"{sample_id}_pc.npy")
            if os.path.exists(cache_file):
                try:
                    if self.debug:
                        print(f"Loading cached point cloud for sample {sample_id}")
                    pc = np.load(cache_file)
                    return torch.tensor(pc, dtype=torch.float32)
                except Exception as e:
                    print(f"Error loading cached point cloud: {e}")
                    # If loading fails, fall through to generation
        
        # Generate point cloud from mesh if we have one
        if mesh is not None:
            try:
                # Sample point cloud from mesh
                pc = sample_points_from_mesh(
                    mesh, 
                    n_points=self.num_points,
                    include_normals=True
                )
                
                # Cache the point cloud
                if cache_file:
                    np.save(cache_file, pc)
                    if self.debug:
                        print(f"Cached point cloud for sample {sample_id}")
                
                return torch.tensor(pc, dtype=torch.float32)
            
            except Exception as e:
                print(f"Error generating point cloud for sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # If we get here, we couldn't load or generate a point cloud
        # print(f"Could not load or generate point cloud for sample {sample_id}")
        return torch.zeros((0, 6), dtype=torch.float32)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample dictionary with mesh/point cloud data, label, and metadata
        """
        try:
            sample = self.samples[idx]
            sample_id = sample['sample_id']
            
            # Base metadata for all return types
            metadata = {
                'sample_id': sample_id,
                'class_name': sample['class_name']
            }
            
            # Return data based on the requested type
            if self.return_type == 'mesh' or self.return_type == 'both':
                # Load or generate mesh
                mesh = self._load_or_generate_mesh(sample)
                
                if mesh is None:
                    # Fallback with empty data
                    metadata['num_vertices'] = 0
                    metadata['num_faces'] = 0
                    vertices = torch.zeros((0, 3), dtype=torch.float32)
                    faces = torch.zeros((0, 3), dtype=torch.long)
                else:
                    # Extract mesh data
                    metadata['num_vertices'] = len(mesh.vertices)
                    metadata['num_faces'] = len(mesh.faces)
                    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                    faces = torch.tensor(mesh.faces, dtype=torch.long)
            
            if self.return_type == 'pointcloud' or self.return_type == 'both':
                # For point cloud, we might need the mesh if not precomputed/cached
                if self.return_type == 'both' and 'mesh' in locals():
                    # We already have the mesh from above, use it
                    pc = self._load_or_generate_pointcloud(sample, mesh=mesh)
                else:
                    # Either we're only returning pointcloud, or we couldn't load a mesh above
                    # Try to load the precomputed or cached point cloud first
                    pc = self._load_or_generate_pointcloud(sample)
                    
                    # If that failed and we need to generate it, load mesh temporarily
                    if pc.shape[0] == 0 and self.generate_on_load:
                        temp_mesh = self._load_or_generate_mesh(sample)
                        if temp_mesh is not None:
                            pc = self._load_or_generate_pointcloud(sample, mesh=temp_mesh)
                
                metadata['num_points'] = pc.shape[0]
            
            # Construct the return dictionary based on the requested type
            result = {
                'label': sample['class_id'],
                'metadata': metadata
            }
            
            if self.return_type == 'mesh':
                result['vertices'] = vertices
                result['faces'] = faces
            elif self.return_type == 'pointcloud':
                # Split the point cloud into positions and normals if available
                if pc.shape[1] == 6:  # Has normals
                    result['points'] = pc[:, :3]
                    result['normals'] = pc[:, 3:]
                else:
                    result['points'] = pc
            
            # Apply point-cloud augmentations if enabled
            if self.apply_aug and pc.shape[0] > 0:
                pts_np = pc[:, :3].numpy()
                # simple transforms
                pts_np = self.pc_augment(pts_np)
                # ARAP / biharmonic warp with prob 0.3
                if random.random() < 0.3:
                    pts_np = arap_warp(pts_np)
                pc[:, :3] = torch.from_numpy(pts_np)
            
            return result
            
        except Exception as e:
            print(f"Error in __getitem__ for idx={idx}: {e}")
            import traceback
            traceback.print_exc()
            raise


def custom_collate_fn(batch):
    """Modified collate function to match MorphoFeatures format"""
    if not batch:
        return {}
    
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'metadata':
            # Handle metadata dictionary
            result[key] = {}
            metadata_keys = batch[0][key].keys()
            
            for metadata_key in metadata_keys:
                result[key][metadata_key] = [sample[key][metadata_key] for sample in batch]
        
        elif key == 'label':
            # Convert labels to tensor
            result[key] = torch.tensor([sample[key] for sample in batch])
        
        elif key == 'vertices':
            # Handle vertices with padding
            max_vertices = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            
            padded_vertices = torch.zeros(batch_size, max_vertices, 3)
            vertex_masks = torch.zeros(batch_size, max_vertices)
            
            for i, sample in enumerate(batch):
                num_vertices = sample[key].shape[0]
                padded_vertices[i, :num_vertices, :] = sample[key]
                vertex_masks[i, :num_vertices] = 1  # Mask to identify real vertices
            
            result[key] = padded_vertices
            result['vertex_masks'] = vertex_masks
        
        elif key == 'faces':
            # Handle faces with padding and adjustment for vertex indices
            max_faces = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            
            padded_faces = torch.ones(batch_size, max_faces, 3, dtype=torch.long) * -1  # -1 for invalid indices
            face_masks = torch.zeros(batch_size, max_faces)
            
            for i, sample in enumerate(batch):
                num_faces = sample[key].shape[0]
                padded_faces[i, :num_faces, :] = sample[key]
                face_masks[i, :num_faces] = 1  # Mask to identify real faces
            
            result[key] = padded_faces
            result['face_masks'] = face_masks
        
        elif key == 'points' or key == 'normals':
            # Handle point data with padding
            max_points = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            feature_dim = batch[0][key].shape[1]
            
            padded_data = torch.zeros(batch_size, max_points, feature_dim)
            data_masks = torch.zeros(batch_size, max_points)
            
            for i, sample in enumerate(batch):
                num_points = sample[key].shape[0]
                padded_data[i, :num_points, :] = sample[key]
                data_masks[i, :num_points] = 1  # Mask to identify real points
            
            result[key] = padded_data
            
            # Create masks only once if they don't exist yet
            if 'point_masks' not in result:
                result['point_masks'] = data_masks
    
    # Add reshaping for MorphoFeatures compatibility
    if 'points' in result and 'normals' in result:
        points = result['points']  # Shape: [B, N, 3]
        normals = result['normals']  # Shape: [B, N, 3]
        
        # Reshape points to [B, 3, N, 1]
        points_reshaped = points.transpose(1, 2).unsqueeze(-1)
        result['points'] = points_reshaped
        
        # Combine points and normals for features: [B, 6, N, 1]
        features = torch.cat([points, normals], dim=2).transpose(1, 2).unsqueeze(-1)
        result['features'] = features
    
    return result


def get_mesh_dataloader_v2(root_dir,
                           batch_size=8,
                           shuffle=True,
                           num_workers=4,
                           class_csv_path=None,
                           sample_ids=None,
                           filter_by_class=None,
                           ignore_unclassified=False,
                           precomputed_dir=None,
                           generate_on_load=True,
                           return_type='mesh',  # 'mesh', 'pointcloud', or 'both'
                           voxel_size=(1.0, 1.0, 1.0),
                           smooth_iterations=10,
                           decimate_target=5000,
                           num_points=1024,
                           sample_percent=100,
                           cache_dir=None,
                           pin_memory=False,
                           debug=False):
    """
    Create a DataLoader for the mesh dataset.
    
    Args:
        root_dir (str): Root directory containing the samples.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes.
        class_csv_path (str, optional): Path to CSV file with class information.
        sample_ids (list, optional): List of sample IDs to include.
        filter_by_class (int or list, optional): Class ID(s) to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        precomputed_dir (str, optional): Directory with pre-processed meshes.
        generate_on_load (bool): Whether to generate meshes on load if not cached/precomputed.
        return_type (str): Type of data to return - 'mesh', 'pointcloud', or 'both'.
        voxel_size (tuple): Size of each voxel (dz, dy, dx) for mesh generation.
        smooth_iterations (int): Number of Taubin smoothing iterations.
        decimate_target (int): Target number of faces after decimation.
        num_points (int): Number of points to sample from mesh surfaces.
        sample_percent (int): Percentage of samples to load per class (1-100).
        cache_dir (str, optional): Directory to cache processed meshes.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        debug (bool): Whether to print debug information.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the mesh dataset.
    """
    dataset = MeshDataset(
        root_dir=root_dir,
        class_csv_path=class_csv_path,
        sample_ids=sample_ids,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        precomputed_dir=precomputed_dir,
        generate_on_load=generate_on_load,
        return_type=return_type,
        voxel_size=voxel_size,
        smooth_iterations=smooth_iterations,
        decimate_target=decimate_target,
        num_points=num_points,
        sample_percent=sample_percent,
        cache_dir=cache_dir,
        debug=debug
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the mesh dataloader")
    parser.add_argument("--root_dir", type=str, default="data/nuclei_sample_1a_v1", 
                        help="Root directory containing the samples")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv",
                        help="Path to CSV file with class information")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with pre-processed meshes")
    parser.add_argument("--return_type", type=str, default="both", choices=["mesh", "pointcloud", "both"],
                        help="Type of data to return")
    parser.add_argument("--sample_id", type=int, default=None,
                        help="Specific sample ID to process")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    
    args = parser.parse_args()
    
    # Prepare sample IDs if specified
    sample_ids = [args.sample_id] if args.sample_id is not None else None
    
    # Create dataloader
    loader = get_mesh_dataloader_v2(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        precomputed_dir=args.precomputed_dir,
        sample_ids=sample_ids,
        batch_size=args.batch_size,
        return_type=args.return_type,
        cache_dir="data/mesh_cache",
        num_workers=0,
        debug=True
    )
    
    print(f"Dataset contains {len(loader.dataset)} samples")
    
    # Process one batch and display information
    for batch in loader:
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
        
        print("\nSample IDs:", batch['metadata']['sample_id'])
        
        break  # Only process one batch 