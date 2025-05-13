import os
import numpy as np
import pandas as pd
import glob
from PIL import Image
import sys
from tqdm import tqdm
import open3d as o3d
from skimage import measure

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def load_mask_volume(mask_dir):
    """
    Load a complete 3D mask volume from a series of TIFF slices.
    
    Args:
        mask_dir (str): Directory containing the mask slices
        
    Returns:
        numpy.ndarray: Mask volume as numpy array with shape (D, H, W)
    """
    # Find all TIFF files in the mask directory
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
    
    if not mask_files:
        raise ValueError(f"No TIFF files found in {mask_dir}")
    
    # Load the first image to get dimensions
    first_mask = np.array(Image.open(mask_files[0]))
    height, width = first_mask.shape
    
    # Create empty volume
    mask_volume = np.zeros((len(mask_files), height, width), dtype=np.float32)
    
    # Load all slices
    print(f"Loading {len(mask_files)} mask slices...")
    for i, mask_file in enumerate(mask_files):
        mask_slice = np.array(Image.open(mask_file), dtype=np.float32)
        
        # Ensure mask is binary
        mask_slice = (mask_slice > 0).astype(np.float32)
        
        mask_volume[i] = mask_slice
    
    return mask_volume

def create_pointcloud_from_mask(mask_volume, threshold=0.5, voxel_size=1.0, sample_rate=1.0, max_points=None):
    """
    Create a point cloud from a binary mask volume.
    
    Args:
        mask_volume (numpy.ndarray): 3D binary mask volume with shape (D, H, W)
        threshold (float): Threshold for considering a voxel as part of the mask
        voxel_size (float): Size of each voxel in the output point cloud
        sample_rate (float): Fraction of points to keep (1.0 = all points)
        max_points (int, optional): Maximum number of points to include in the point cloud
        
    Returns:
        open3d.geometry.PointCloud: Point cloud object
    """
    # Find coordinates of non-zero voxels (these are our points)
    points = np.column_stack(np.where(mask_volume > threshold))
    
    # Apply max_points limit if specified (takes precedence over sample_rate)
    if max_points is not None and len(points) > max_points:
        # Calculate the required sample rate to achieve max_points
        required_sample_rate = max_points / len(points)
        # Use the smaller of the two rates
        effective_sample_rate = min(sample_rate, required_sample_rate)
    else:
        effective_sample_rate = sample_rate
    
    # Sample points if needed
    if effective_sample_rate < 1.0:
        num_points = len(points)
        num_samples = max(1, int(num_points * effective_sample_rate))
        indices = np.random.choice(num_points, num_samples, replace=False)
        points = points[indices]
    
    print(f"Point cloud contains {len(points)} points")
    
    # Scale points by voxel size
    points = points * voxel_size
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    # Assign colors based on position (for visualization)
    # Normalize coordinates to [0, 1] range for RGB color
    if len(points) > 0:
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        range_coords = max_coords - min_coords
        
        # Avoid division by zero
        range_coords[range_coords == 0] = 1
        
        # Normalize coordinates to [0, 1] for RGB colors
        colors = (points - min_coords) / range_coords
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def create_mesh_from_mask(mask_volume, threshold=0.5, voxel_size=1.0, smoothing_iterations=0):
    """
    Create a triangle mesh from a binary mask volume using marching cubes algorithm.
    
    Args:
        mask_volume (numpy.ndarray): 3D binary mask volume with shape (D, H, W)
        threshold (float): Threshold for considering a voxel as part of the mask
        voxel_size (float): Size of each voxel in the output mesh
        smoothing_iterations (int): Number of iterations for Laplacian smoothing (0 for no smoothing)
        
    Returns:
        open3d.geometry.TriangleMesh: Triangle mesh object
    """
    # Run marching cubes algorithm on the volume
    verts, faces, normals, values = measure.marching_cubes(mask_volume, level=threshold)
    
    # Scale vertices by voxel size
    verts = verts * voxel_size
    
    # Create Open3D triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Compute vertex normals for proper rendering
    mesh.compute_vertex_normals()
    
    # Apply Laplacian smoothing if requested
    if smoothing_iterations > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iterations)
        mesh.compute_vertex_normals()
    
    # Assign colors based on position (for visualization)
    # Normalize coordinates to [0, 1] range for RGB color
    if len(verts) > 0:
        min_coords = np.min(verts, axis=0)
        max_coords = np.max(verts, axis=0)
        range_coords = max_coords - min_coords
        
        # Avoid division by zero
        range_coords[range_coords == 0] = 1
        
        # Normalize coordinates to [0, 1] for RGB colors
        colors = (verts - min_coords) / range_coords
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Mesh contains {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    return mesh

def process_sample(sample_id, root_dir, output_dir, voxel_size=1.0, sample_rate=0.5, max_points=None):
    """
    Process a single sample: load mask volume, create point cloud, and save to file.
    
    Args:
        sample_id (str): ID of the sample to process
        root_dir (str): Root directory containing all samples
        output_dir (str): Directory to save the point cloud
        voxel_size (float): Size of each voxel in the output point cloud
        sample_rate (float): Fraction of points to keep (1.0 = all points)
        max_points (int, optional): Maximum number of points to include in the point cloud
        
    Returns:
        str: Path to the saved point cloud file, or None if processing failed
    """
    # Format sample ID with leading zeros to match directory structure
    sample_id_str = str(sample_id).zfill(4)
    
    # Find the sample directory
    sample_dirs = glob.glob(os.path.join(root_dir, '*', sample_id_str))
    
    if not sample_dirs:
        print(f"Sample {sample_id} not found in {root_dir}")
        return None
    
    sample_dir = sample_dirs[0]
    mask_dir = os.path.join(sample_dir, 'mask')
    
    try:
        # Load the mask volume
        mask_volume = load_mask_volume(mask_dir)
        
        # Create point cloud
        pcd = create_pointcloud_from_mask(
            mask_volume, 
            threshold=0.5, 
            voxel_size=voxel_size,
            sample_rate=sample_rate,
            max_points=max_points
        )
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{sample_id_str}.ply")
        
        # Save point cloud
        o3d.io.write_point_cloud(output_file, pcd)
        
        print(f"Saved point cloud for sample {sample_id} to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return None

def process_sample_mesh(sample_id, root_dir, output_dir, voxel_size=1.0, smoothing_iterations=1):
    """
    Process a single sample: load mask volume, create triangle mesh, and save to file.
    
    Args:
        sample_id (str): ID of the sample to process
        root_dir (str): Root directory containing all samples
        output_dir (str): Directory to save the mesh
        voxel_size (float): Size of each voxel in the output mesh
        smoothing_iterations (int): Number of iterations for Laplacian smoothing
        
    Returns:
        str: Path to the saved mesh file, or None if processing failed
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
        # Load the mask volume
        mask_volume = load_mask_volume(mask_dir)
        
        # Create mesh
        mesh = create_mesh_from_mask(
            mask_volume, 
            threshold=0.5, 
            voxel_size=voxel_size,
            smoothing_iterations=smoothing_iterations
        )
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{sample_id_str}.obj")
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_file, mesh)
        
        print(f"Saved mesh for sample {sample_id} to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_samples(sample_dir, output_dir, voxel_size=1.0, sample_rate=0.5, max_points=None):
    """
    Process all samples in the directory.
    
    Args:
        sample_dir (str): Directory containing all samples
        output_dir (str): Directory to save the point clouds
        voxel_size (float): Size of each voxel in the output point cloud
        sample_rate (float): Fraction of points to keep (1.0 = all points)
        max_points (int, optional): Maximum number of points to include in the point cloud
        
    Returns:
        dict: Dictionary mapping sample IDs to point cloud file paths
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all sample directories
    sample_dirs = glob.glob(os.path.join(sample_dir, '*', '[0-9][0-9][0-9][0-9]'))
    sample_ids = [os.path.basename(d) for d in sample_dirs]
    
    if not sample_ids:
        print(f"No sample directories found in {sample_dir}")
        return {}
    
    results = {}
    
    # Process each sample
    for sample_id in tqdm(sample_ids, desc="Processing samples"):
        output_file = process_sample(
            sample_id, 
            sample_dir, 
            output_dir, 
            voxel_size=voxel_size,
            sample_rate=sample_rate,
            max_points=max_points
        )
        
        if output_file:
            results[sample_id] = output_file
    
    print(f"Processed {len(results)} samples out of {len(sample_ids)}")
    return results

def main():
    """
    Main function to run the point cloud generation process.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D point clouds from masked nuclei samples")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Root directory containing all samples")
    parser.add_argument("--output_dir", type=str, default="data/pointclouds", 
                        help="Directory to save the point clouds")
    parser.add_argument("--voxel_size", type=float, default=1.0, 
                        help="Size of each voxel in the output point cloud")
    parser.add_argument("--sample_rate", type=float, default=0.1, 
                        help="Fraction of points to keep (1.0 = all points)")
    parser.add_argument("--max_points", type=int, default=10000, 
                        help="Maximum number of points to include in the point cloud")
    parser.add_argument("--sample_id", type=str, default=None,
                        help="Process only a specific sample ID")
    
    args = parser.parse_args()
    
    # Process a single sample if specified
    if args.sample_id:
        process_sample(
            args.sample_id, 
            args.data_dir, 
            args.output_dir, 
            voxel_size=args.voxel_size,
            sample_rate=args.sample_rate,
            max_points=args.max_points
        )
    else:
        # Process all samples
        process_all_samples(
            args.data_dir, 
            args.output_dir, 
            voxel_size=args.voxel_size,
            sample_rate=args.sample_rate,
            max_points=args.max_points
        )

if __name__ == "__main__":
    main() 