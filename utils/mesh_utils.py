import os
import numpy as np
import glob
from PIL import Image
import sys
import open3d as o3d
from skimage import measure
import trimesh
import pathlib

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.pointcloud import load_mask_volume

def create_mesh_from_mask(mask_volume, threshold=0.5, voxel_size=(1.0, 1.0, 1.0), 
                         smooth_iterations=10, decimate_target=5000):
    """
    Create a triangle mesh from a binary mask volume using marching cubes algorithm.
    
    Args:
        mask_volume (numpy.ndarray): 3D binary mask volume with shape (D, H, W)
        threshold (float): Threshold for considering a voxel as part of the mask
        voxel_size (tuple): Size of each voxel in the output mesh (dz, dy, dx)
        smooth_iterations (int): Number of iterations for Taubin smoothing
        decimate_target (int): Target number of faces after decimation
        
    Returns:
        trimesh.Trimesh: Triangle mesh object
    """
    # Step 1: Run marching cubes algorithm on the volume
    verts, faces, normals, _ = measure.marching_cubes(mask_volume, level=threshold, spacing=voxel_size)
    
    # Create trimesh mesh
    raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Apply Taubin smoothing if requested
    if smooth_iterations > 0:
        try:
            smoothed_mesh = raw_mesh.copy()
            smoothed_mesh = smoothed_mesh.smoothed(filter='taubin', iterations=smooth_iterations, 
                                                 lamb=0.5, nu=-0.53)
        except Exception as e:
            print(f"Warning: Smoothing failed: {e}. Using raw mesh instead.")
            smoothed_mesh = raw_mesh
    else:
        smoothed_mesh = raw_mesh
    
    # Decimation if requested - fixed to use pymeshlab if available or fallback
    if decimate_target > 0 and decimate_target < len(smoothed_mesh.faces):
        try:
            # Try using quadric decimation directly if available
            if hasattr(trimesh.simplify, 'simplify_quadratic_decimation'):
                decimated_mesh = trimesh.simplify.simplify_quadratic_decimation(
                    smoothed_mesh, decimate_target)
            # Otherwise try pymeshlab if available
            else:
                try:
                    import pymeshlab
                    # Create a new MeshSet
                    ms = pymeshlab.MeshSet()
                    # Convert trimesh to pymeshlab format
                    mesh_data = pymeshlab.Mesh(smoothed_mesh.vertices, smoothed_mesh.faces)
                    # Add to the MeshSet
                    ms.add_mesh(mesh_data)
                    # Simplify the mesh
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=decimate_target)
                    # Get the mesh data back
                    decimated_vertices = ms.current_mesh().vertex_matrix()
                    decimated_faces = ms.current_mesh().face_matrix()
                    # Create a new trimesh from the decimated data
                    decimated_mesh = trimesh.Trimesh(vertices=decimated_vertices, 
                                                     faces=decimated_faces)
                except ImportError:
                    print(f"Warning: Neither pymeshlab nor trimesh.simplify available. Skipping decimation.")
                    decimated_mesh = smoothed_mesh
        except Exception as e:
            print(f"Warning: Decimation failed: {e}. Using smoothed mesh instead.")
            decimated_mesh = smoothed_mesh
    else:
        decimated_mesh = smoothed_mesh
    
    # Make sure the mesh is watertight and has correct face winding
    try:
        # Fill holes
        decimated_mesh.fill_holes()
        # Fix face winding and normals
        decimated_mesh.fix_normals()
    except Exception as e:
        print(f"Warning: Mesh repair operations failed: {e}")
    
    print(f"Mesh contains {len(decimated_mesh.vertices)} vertices and {len(decimated_mesh.faces)} faces")
    return decimated_mesh

def sample_points_from_mesh(mesh, n_points=1024, include_normals=True):
    """
    Sample a point cloud from a mesh surface with even distribution.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        n_points (int): Number of points to sample
        include_normals (bool): Whether to include normals in the output
        
    Returns:
        numpy.ndarray: Point cloud with shape (n_points, 3) or (n_points, 6) with normals
    """
    try:
        # Sample points evenly across the surface using better sampling method
        points, face_indices = mesh.sample(n_points, return_index=True)
        
        if include_normals:
            # Get normals from the faces
            normals = mesh.face_normals[face_indices]
            point_cloud = np.hstack([points, normals]).astype(np.float32)
        else:
            point_cloud = points.astype(np.float32)
            
        # Ensure we have exactly n_points
        if len(points) < n_points:
            # If we didn't get enough points, duplicate some randomly
            print(f"only got {len(points)}/{n_points} samples!")
            extra_needed = n_points - len(points)
            if len(points) > 0:  # Only if we have at least some points
                # Duplicate some points randomly
                indices = np.random.choice(len(points), extra_needed, replace=True)
                extra_points = point_cloud[indices]
                point_cloud = np.vstack([point_cloud, extra_points])
            
        return point_cloud
    except Exception as e:
        print(f"Warning: Point sampling failed: {e}. Falling back to random sampling.")
        # Fallback to random vertices with their vertex normals
        indices = np.random.choice(
            len(mesh.vertices), min(n_points, len(mesh.vertices)), replace=len(mesh.vertices) < n_points)
        points = mesh.vertices[indices]
        
        if include_normals:
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
                normals = mesh.vertex_normals[indices]
            else:
                normals = np.zeros_like(points)
            point_cloud = np.hstack([points, normals]).astype(np.float32)
        else:
            point_cloud = points.astype(np.float32)
        
        # Ensure we have exactly n_points
        if len(point_cloud) < n_points:
            extra_needed = n_points - len(point_cloud)
            # Duplicate some points randomly if we have at least one point
            if len(point_cloud) > 0:
                indices = np.random.choice(len(point_cloud), extra_needed, replace=True)
                extra_points = point_cloud[indices]
                point_cloud = np.vstack([point_cloud, extra_points])
            
        return point_cloud

def process_sample_mesh(sample_id, root_dir, output_dir, voxel_size=(1.0, 1.0, 1.0), 
                       smooth_iterations=10, decimate_target=5000, sample_points=1024):
    """
    Process a single sample: load mask volume, create triangle mesh, sample point cloud, and save.
    
    Args:
        sample_id (str): ID of the sample to process
        root_dir (str): Root directory containing all samples
        output_dir (str): Directory to save the mesh and point cloud
        voxel_size (tuple): Size of each voxel in the output mesh (dz, dy, dx)
        smooth_iterations (int): Number of iterations for Taubin smoothing
        decimate_target (int): Target number of faces after decimation
        sample_points (int): Number of points to sample from the mesh surface
        
    Returns:
        dict: Dictionary with paths to the saved mesh and point cloud files
    """
    # Format sample ID with leading zeros to match directory structure
    sample_id_str = str(sample_id).zfill(4)
    
    # Find the sample directory
    sample_dir = os.path.join(root_dir, sample_id_str)
    mask_dir = os.path.join(sample_dir, 'mask')
    
    if not os.path.exists(mask_dir):
        print(f"Mask directory not found for sample {sample_id}: {mask_dir}")
        return None
    
    try:
        # Create output directories
        mesh_dir = os.path.join(output_dir, "meshes")
        pc_dir = os.path.join(output_dir, "pointclouds")
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(pc_dir, exist_ok=True)
        
        # Load the mask volume
        mask_volume = load_mask_volume(mask_dir)
        
        # Create mesh
        mesh = create_mesh_from_mask(
            mask_volume, 
            threshold=0.5, 
            voxel_size=voxel_size,
            smooth_iterations=smooth_iterations,
            decimate_target=decimate_target
        )
        
        # Sample point cloud
        point_cloud = sample_points_from_mesh(mesh, n_points=sample_points)
        
        # Create output filenames
        formatted_id = str(sample_id).zfill(6)
        mesh_path = os.path.join(mesh_dir, f"{formatted_id}_mesh.ply")
        pc_path = os.path.join(pc_dir, f"{formatted_id}_pc.npy")
        
        # Save mesh
        mesh.export(mesh_path)
        
        # Save point cloud
        np.save(pc_path, point_cloud)
        
        print(f"Processed sample {sample_id}:")
        print(f"  - Mesh saved to: {mesh_path}")
        print(f"  - Point cloud saved to: {pc_path}")
        
        return {
            'sample_id': sample_id,
            'mesh_path': mesh_path,
            'pc_path': pc_path,
            'mesh': mesh,
            'point_cloud': point_cloud
        }
    
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_samples(root_dir, output_dir, voxel_size=(1.0, 1.0, 1.0), 
                       smooth_iterations=10, decimate_target=5000, sample_points=1024,
                       sample_ids=None, max_samples=None):
    """
    Process all samples in the directory or specified sample IDs.
    
    Args:
        root_dir (str): Root directory containing all samples
        output_dir (str): Directory to save the meshes and point clouds
        voxel_size (tuple): Size of each voxel in the output mesh (dz, dy, dx)
        smooth_iterations (int): Number of iterations for Taubin smoothing
        decimate_target (int): Target number of faces after decimation
        sample_points (int): Number of points to sample from each mesh surface
        sample_ids (list, optional): List of specific sample IDs to process
        max_samples (int, optional): Maximum number of samples to process
        
    Returns:
        dict: Dictionary mapping sample IDs to processing results
    """
    # Find all sample directories if specific IDs not provided
    if sample_ids is None:
        sample_dirs = []
        try:
            for item in os.listdir(root_dir):
                if os.path.isdir(os.path.join(root_dir, item)) and item.isdigit() and len(item) == 4:
                    # Check if it has a mask subdirectory
                    mask_dir = os.path.join(root_dir, item, 'mask')
                    if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                        sample_dirs.append(item)
        except Exception as e:
            print(f"Error scanning directory {root_dir}: {e}")
            return {}
        
        # Convert to sample IDs
        sample_ids = [int(d) for d in sample_dirs]
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples > 0:
        sample_ids = sample_ids[:max_samples]
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample
    results = {}
    for sample_id in sample_ids:
        print(f"\nProcessing sample {sample_id}...")
        result = process_sample_mesh(
            sample_id, 
            root_dir, 
            output_dir,
            voxel_size=voxel_size,
            smooth_iterations=smooth_iterations,
            decimate_target=decimate_target,
            sample_points=sample_points
        )
        
        if result:
            results[sample_id] = result
    
    print(f"\nProcessed {len(results)} samples out of {len(sample_ids)}")
    return results

def main():
    """
    Main function to run the mesh generation process.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D meshes and point clouds from masked nuclei samples")
    parser.add_argument("--data_dir", type=str, default="data/nuclei_sample_1a_v1", 
                        help="Root directory containing all samples")
    parser.add_argument("--output_dir", type=str, default="data/mesh_processed", 
                        help="Directory to save the meshes and point clouds")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0], 
                        help="Size of each voxel in the output mesh (dz, dy, dx)")
    parser.add_argument("--smooth_iterations", type=int, default=10, 
                        help="Number of iterations for Taubin smoothing")
    parser.add_argument("--decimate_target", type=int, default=5000, 
                        help="Target number of faces after decimation")
    parser.add_argument("--sample_points", type=int, default=1024, 
                        help="Number of points to sample from each mesh surface")
    parser.add_argument("--sample_id", type=int, default=None,
                        help="Process only a specific sample ID")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    # Process a single sample if specified
    if args.sample_id is not None:
        process_sample_mesh(
            args.sample_id, 
            args.data_dir, 
            args.output_dir, 
            voxel_size=tuple(args.voxel_size),
            smooth_iterations=args.smooth_iterations,
            decimate_target=args.decimate_target,
            sample_points=args.sample_points
        )
    else:
        # Process all samples (or up to max_samples)
        process_all_samples(
            args.data_dir, 
            args.output_dir, 
            voxel_size=tuple(args.voxel_size),
            smooth_iterations=args.smooth_iterations,
            decimate_target=args.decimate_target,
            sample_points=args.sample_points,
            max_samples=args.max_samples
        )

if __name__ == "__main__":
    main() 