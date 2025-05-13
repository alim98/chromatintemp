import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.mesh_utils import create_mesh_from_mask, sample_points_from_mesh
from utils.pointcloud import load_mask_volume, create_pointcloud_from_mask

def visualize_comparison(sample_id, smooth_iterations=10, decimate_target=5000, num_points=1024,
                        voxel_size=(1.0, 1.0, 1.0), save_html=False, show_mesh=True, show_pointcloud=True):
    """
    Visualize a 3D mesh and point cloud comparison from a mask volume.
    
    Args:
        sample_id (int): ID of the sample to visualize
        smooth_iterations (int): Number of iterations for Taubin smoothing
        decimate_target (int): Target number of faces after decimation
        num_points (int): Number of points to sample from the mesh surface
        voxel_size (tuple): Size of each voxel (dz, dy, dx)
        save_html (bool): Whether to save the visualization as HTML file
        show_mesh (bool): Whether to show the mesh visualization
        show_pointcloud (bool): Whether to show the point cloud visualization
    """
    # Format sample ID with leading zeros
    sample_id_str = str(sample_id).zfill(4)
    
    # Find the sample directory
    sample_dir = os.path.join(config.DATA_ROOT, sample_id_str)
    mask_dir = os.path.join(sample_dir, 'mask')
    
    if not os.path.exists(mask_dir):
        print(f"Mask directory not found for sample {sample_id}: {mask_dir}")
        return
    
    print(f"Processing sample {sample_id} from {mask_dir}")
    
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
    
    # Sample point cloud from mesh
    point_cloud_from_mesh = None
    if show_pointcloud:
        point_cloud_from_mesh = sample_points_from_mesh(
            mesh, 
            n_points=num_points, 
            include_normals=True
        )
    
    # Create visualization
    if show_mesh or show_pointcloud:
        # Determine number of subplots
        if show_mesh and show_pointcloud:
            fig = make_subplots(
                rows=1, cols=2, 
                specs=[[{'type': 'surface'}, {'type': 'scatter3d'}]],
                subplot_titles=("Triangle Mesh", "Point Cloud")
            )
        else:
            fig = make_subplots(
                rows=1, cols=1, 
                specs=[[{'type': 'surface' if show_mesh else 'scatter3d'}]]
            )
        
        # Add mesh visualization
        if show_mesh:
            # Extract vertices and faces from trimesh
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            # Create mesh visualization
            mesh_vis = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale='Viridis',
                opacity=0.8,
                intensity=vertices[:, 0] + vertices[:, 1] + vertices[:, 2],
                showscale=False
            )
            
            fig.add_trace(mesh_vis, row=1, col=1)
        
        # Add point cloud visualization
        if show_pointcloud:
            if point_cloud_from_mesh is not None:
                points = point_cloud_from_mesh[:, :3]  # First 3 columns are points
                normals = point_cloud_from_mesh[:, 3:] if point_cloud_from_mesh.shape[1] > 3 else None
                
                # Create point cloud visualization
                point_vis = go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=points[:, 2],  # Color by z-coordinate
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name="Point Cloud"
                )
                
                col_idx = 2 if show_mesh else 1
                fig.add_trace(point_vis, row=1, col=col_idx)
        
        # Update layout
        fig.update_layout(
            title=f"Sample {sample_id} Visualization",
            scene=dict(
                aspectmode='data'
            ),
            scene2=dict(
                aspectmode='data'
            ) if show_mesh and show_pointcloud else None,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save visualization if requested
        if save_html:
            os.makedirs(os.path.join(config.RESULTS_DIR, "html_pointcloud"), exist_ok=True)
            output_path = os.path.join(config.RESULTS_DIR, "html_pointcloud", f"sample_{sample_id_str}.html")
            fig.write_html(output_path)
            print(f"Visualization saved to {output_path}")
        
        # Show interactive visualization
        fig.show()
    
    return mesh, point_cloud_from_mesh

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D nuclei samples as mesh and point cloud")
    parser.add_argument("--sample_id", type=int, required=True, help="Sample ID to visualize")
    parser.add_argument("--smooth", type=int, default=10, help="Number of smoothing iterations")
    parser.add_argument("--decimate", type=int, default=5000, help="Target number of faces after decimation")
    parser.add_argument("--points", type=int, default=1024, help="Number of points to sample from mesh")
    parser.add_argument("--save", action="store_true", help="Save visualization as HTML")
    parser.add_argument("--mesh_only", action="store_true", help="Show only mesh visualization")
    parser.add_argument("--point_only", action="store_true", help="Show only point cloud visualization")
    
    args = parser.parse_args()
    
    # Determine what to show
    show_mesh = not args.point_only
    show_pointcloud = not args.mesh_only
    
    visualize_comparison(
        sample_id=args.sample_id,
        smooth_iterations=args.smooth,
        decimate_target=args.decimate,
        num_points=args.points,
        save_html=args.save,
        show_mesh=show_mesh,
        show_pointcloud=show_pointcloud
    )

if __name__ == "__main__":
    main() 