import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataloader.mesh_dataloader import get_mesh_dataloader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
import trimesh
import pathlib
from utils.pointcloud import load_mask_volume
import plotly.graph_objects as go
import plotly.offline as pyo

# Load mask volume and create mesh using MorphoFeatures-like pipeline
def create_mesh_from_mask_morphofeatures(mask_volume, voxel_size=(1.0, 1.0, 1.0), 
                                        smooth_iterations=10, decimate_target=5000,
                                        sample_points=1024):
    """
    Create and process a mesh from binary mask volume following MorphoFeatures approach.
    
    Args:
        mask_volume (numpy.ndarray): 3D binary mask volume with shape (D, H, W)
        voxel_size (tuple): Size of each voxel in 3D space (dz, dy, dx)
        smooth_iterations (int): Number of iterations for Taubin smoothing
        decimate_target (int): Target number of faces after decimation
        sample_points (int): Number of points to sample from the mesh surface
        
    Returns:
        tuple: (raw_mesh, smoothed_mesh, decimated_mesh, point_cloud)
    """
    print(f"Creating mesh from mask volume of shape {mask_volume.shape}...")
    
    # Step 1: Create raw surface using marching cubes
    print("Step 1: Generating raw surface with marching cubes...")
    verts, faces, normals, _ = marching_cubes(mask_volume, level=0.5, spacing=voxel_size)
    raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    print(f"Raw mesh has {len(raw_mesh.vertices)} vertices and {len(raw_mesh.faces)} faces")
    
    # Step 2: Apply Taubin smoothing
    print(f"Step 2: Applying Taubin smoothing with {smooth_iterations} iterations...")
    smoothed_mesh = raw_mesh.copy()
    try:
        smoothed_mesh = smoothed_mesh.smoothed(filter='taubin', iterations=smooth_iterations, lamb=0.5, nu=-0.53)
        print(f"Smoothed mesh has {len(smoothed_mesh.vertices)} vertices and {len(smoothed_mesh.faces)} faces")
    except Exception as e:
        print(f"Warning: Smoothing failed: {e}. Using raw mesh instead.")
        smoothed_mesh = raw_mesh
    
    # Step 3: Decimation and repair
    print(f"Step 3: Decimating to target {decimate_target} faces...")
    decimated_mesh = smoothed_mesh.copy()
    try:
        # Perform decimation (use trimesh's built-in decimation as a fallback)
        # For more advanced decimation and repair, pymeshlab would be needed
        if len(smoothed_mesh.faces) > decimate_target:
            decimated_mesh = trimesh.simplify.simplify_quadratic_decimation(smoothed_mesh, decimate_target)
            print(f"Decimated mesh has {len(decimated_mesh.vertices)} vertices and {len(decimated_mesh.faces)} faces")
    except Exception as e:
        print(f"Warning: Decimation failed: {e}. Using smoothed mesh instead.")
        decimated_mesh = smoothed_mesh
    
    # Step 4: Sample a uniform point cloud from the surface
    print(f"Step 4: Sampling {sample_points} points from the surface...")
    try:
        points, face_indices = trimesh.sample.sample_surface_even(decimated_mesh, sample_points)
        normals = decimated_mesh.face_normals[face_indices]
        point_cloud = np.hstack([points, normals]).astype(np.float32)
        print(f"Point cloud has shape {point_cloud.shape}")
    except Exception as e:
        print(f"Warning: Point cloud sampling failed: {e}. Using random points.")
        points = decimated_mesh.vertices[np.random.choice(
            len(decimated_mesh.vertices), min(sample_points, len(decimated_mesh.vertices)), replace=False)]
        normals = np.zeros_like(points)
        point_cloud = np.hstack([points, normals]).astype(np.float32)
    
    return raw_mesh, smoothed_mesh, decimated_mesh, point_cloud

# Function to save mesh visualization as HTML using Plotly
def save_mesh_plotly(mesh, output_path, title="Mesh", color='red', opacity=0.7):
    """
    Save a mesh visualization as an interactive HTML file using Plotly
    
    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        output_path (str): Path to save the HTML file
        title (str): Title for the visualization
        color (str): Color of the mesh
        opacity (float): Opacity of the mesh
    """
    # Create mesh visualization
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create x, y, z coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Convert faces to the format expected by Plotly
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # Create mesh3d trace
    mesh_trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=opacity,
        color=color,
        flatshading=True,
        name=title
    )
    
    # Create figure
    fig = go.Figure(data=[mesh_trace])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Save to HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"Saved mesh visualization to {output_path}")
    
    return output_path

# Function to save point cloud visualization as HTML using Plotly
def save_pointcloud_plotly(points, output_path, title="Point Cloud", color='red', size=2):
    """
    Save a point cloud visualization as an interactive HTML file using Plotly
    
    Args:
        points (numpy.ndarray): Point cloud data (N, 3) or (N, 6) with normals
        output_path (str): Path to save the HTML file
        title (str): Title for the visualization
        color (str): Color of the points
        size (float): Size of the points
    """
    # If points include normals (shape[1] > 3), just use the position part
    if points.shape[1] > 3:
        pos = points[:, :3]
    else:
        pos = points
    
    # Create scatter3d trace
    point_trace = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=0.8
        ),
        name=title
    )
    
    # Create figure
    fig = go.Figure(data=[point_trace])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Save to HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"Saved point cloud visualization to {output_path}")
    
    return output_path

# Function to save processed mesh and point cloud
def save_processed_data(sample_id, raw_mesh, smoothed_mesh, decimated_mesh, point_cloud, output_dir="results"):
    """
    Save processed mesh and point cloud data following MorphoFeatures approach
    """
    # Create output directories
    mesh_dir = os.path.join(output_dir, "meshes")
    pc_dir = os.path.join(output_dir, "pointclouds")
    html_dir = os.path.join(output_dir, "html")
    
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    
    # Save meshes at different processing stages
    sample_id_str = str(sample_id).zfill(6)
    raw_mesh.export(os.path.join(mesh_dir, f"{sample_id_str}_raw.ply"))
    smoothed_mesh.export(os.path.join(mesh_dir, f"{sample_id_str}_smoothed.ply"))
    decimated_mesh.export(os.path.join(mesh_dir, f"{sample_id_str}_mesh.ply"))
    
    # Save point cloud
    np.save(os.path.join(pc_dir, f"{sample_id_str}_pc.npy"), point_cloud)
    
    # Save HTML visualizations
    save_mesh_plotly(raw_mesh, os.path.join(html_dir, f"{sample_id_str}_raw_mesh.html"), 
                    title=f"Raw Mesh - Sample {sample_id}", color='blue', opacity=0.6)
    
    save_mesh_plotly(smoothed_mesh, os.path.join(html_dir, f"{sample_id_str}_smoothed_mesh.html"), 
                    title=f"Smoothed Mesh - Sample {sample_id}", color='green', opacity=0.6)
    
    save_mesh_plotly(decimated_mesh, os.path.join(html_dir, f"{sample_id_str}_final_mesh.html"), 
                    title=f"Final Mesh - Sample {sample_id}", color='red', opacity=0.6)
    
    save_pointcloud_plotly(point_cloud[:, :3], os.path.join(html_dir, f"{sample_id_str}_pointcloud.html"), 
                          title=f"Point Cloud - Sample {sample_id}", color='purple', size=3)
    
    print(f"Saved processed data for sample {sample_id_str}:")
    print(f"  - Raw mesh: {os.path.join(mesh_dir, f'{sample_id_str}_raw.ply')}")
    print(f"  - Smoothed mesh: {os.path.join(mesh_dir, f'{sample_id_str}_smoothed.ply')}")
    print(f"  - Final mesh: {os.path.join(mesh_dir, f'{sample_id_str}_mesh.ply')}")
    print(f"  - Point cloud: {os.path.join(pc_dir, f'{sample_id_str}_pc.npy')}")
    print(f"  - HTML visualizations saved in: {html_dir}")

# Find available sample directories
def find_sample_dirs(root_dir):
    sample_dirs = []
    try:
        for item in os.listdir(root_dir):
            # Check if it's a directory with a 4-digit name
            if os.path.isdir(os.path.join(root_dir, item)) and item.isdigit() and len(item) == 4:
                # Check if it has a mask subdirectory
                mask_dir = os.path.join(root_dir, item, 'mask')
                if os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
                    sample_dirs.append(item)
    except Exception as e:
        print(f"Error scanning directory {root_dir}: {e}")
    
    return sample_dirs

# Main execution
if __name__ == "__main__":
    print("Starting mesh generation process...")
    
    # Parameters
    root_dir = "data/nuclei_sample_1a_v1"
    sample_id = None  # Will be set from available samples
    voxel_size = (1.0, 1.0, 1.0)  # Voxel spacing in 3D space
    smooth_iterations = 10  # Number of Taubin smoothing iterations
    decimate_target = 5000  # Target number of faces after decimation
    sample_points = 1024  # Number of points to sample from surface
    output_dir = "results"  # Output directory for saving data
    
    # Find available samples
    sample_dirs = find_sample_dirs(root_dir)
    
    if not sample_dirs:
        print(f"No sample directories found in {root_dir}!")
        print("Please check the path and ensure there are valid samples with mask directories.")
        sys.exit(1)
    
    print(f"Found {len(sample_dirs)} sample directories: {sample_dirs[:5]}{'...' if len(sample_dirs) > 5 else ''}")
    
    # Use the first sample directory found
    sample_id = int(sample_dirs[0])
    sample_id_str = str(sample_id).zfill(4)
    print(f"Using sample {sample_id_str}")
    
    try:
        # Find the sample directory
        sample_dir = os.path.join(root_dir, sample_id_str)
        mask_dir = os.path.join(sample_dir, 'mask')
        
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory not found: {mask_dir}")
        
        # Load the mask volume
        mask_volume = load_mask_volume(mask_dir)
        print(f"Loaded mask volume of shape {mask_volume.shape}")
        
        # Process the mask volume to create meshes
        raw_mesh, smoothed_mesh, decimated_mesh, point_cloud = create_mesh_from_mask_morphofeatures(
            mask_volume, 
            voxel_size=voxel_size,
            smooth_iterations=smooth_iterations,
            decimate_target=decimate_target,
            sample_points=sample_points
        )
        
        # Save processed data and HTML visualizations
        save_processed_data(
            sample_id, 
            raw_mesh, 
            smoothed_mesh, 
            decimated_mesh, 
            point_cloud, 
            output_dir=output_dir
        )
        
        print("\nProcessing complete!")
        print(f"All data and visualizations have been saved to the {output_dir} directory.")
        print(f"You can open the HTML files in a web browser to interact with the 3D visualizations.")
        
    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()











