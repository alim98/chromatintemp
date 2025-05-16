import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_sphere_mesh(n_vertices=1000, radius=1.0, noise_level=0.05):
    """
    Generate a spherical mesh with the given number of vertices.
    
    Args:
        n_vertices: Number of vertices in the mesh
        radius: Radius of the sphere
        noise_level: Level of noise to add to vertices (0.0 - 1.0)
        
    Returns:
        dict: Dictionary with 'points' and 'faces' tensors
    """
    # Generate points on a sphere
    indices = np.arange(0, n_vertices, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_vertices)
    theta = np.pi * (1 + 5**0.5) * indices
    
    # Convert to Cartesian coordinates
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    
    # Add some noise
    if noise_level > 0:
        x += np.random.normal(0, noise_level * radius, n_vertices)
        y += np.random.normal(0, noise_level * radius, n_vertices)
        z += np.random.normal(0, noise_level * radius, n_vertices)
    
    # Combine points
    points = np.column_stack([x, y, z])
    
    # Generate faces using Delaunay triangulation
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    faces = tri.simplices
    
    # Keep only outward-facing triangles (simple heuristic)
    centers = np.mean(points[faces], axis=1)
    norms = np.sqrt(np.sum(centers**2, axis=1))
    valid_faces = faces[norms > 0.7 * radius]
    
    # Convert to tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    faces_tensor = torch.tensor(valid_faces, dtype=torch.int64)
    
    # Create random "features" for each point (e.g., curvature)
    features = torch.tensor(np.column_stack([
        np.random.normal(0, 0.1, n_vertices),  # Curvature
        np.random.normal(0, 0.1, n_vertices),  # Normal X
        np.random.normal(0, 0.1, n_vertices)   # Normal Y
    ]), dtype=torch.float32)
    
    return {
        'points': points_tensor.unsqueeze(0),  # Add batch dimension
        'features': features.unsqueeze(0),     # Add batch dimension
        'faces': faces_tensor.unsqueeze(0)     # Add batch dimension
    }

def generate_cell_like_mesh(n_vertices=1000, noise_level=0.2):
    """
    Generate a more cell-like mesh with the given number of vertices.
    
    Args:
        n_vertices: Number of vertices in the mesh
        noise_level: Level of noise to add to vertices (0.0 - 1.0)
        
    Returns:
        dict: Dictionary with 'points' and 'faces' tensors
    """
    # Generate points on an ellipsoid
    indices = np.arange(0, n_vertices, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_vertices)
    theta = np.pi * (1 + 5**0.5) * indices
    
    # Use different radii for each axis to create an ellipsoid
    rx, ry, rz = 1.0, 0.8, 0.6
    
    # Convert to Cartesian coordinates
    x = rx * np.cos(theta) * np.sin(phi)
    y = ry * np.sin(theta) * np.sin(phi)
    z = rz * np.cos(phi)
    
    # Add some protrusions to make it cell-like
    n_protrusions = 5
    for i in range(n_protrusions):
        # Random direction
        p_theta = np.random.uniform(0, 2*np.pi)
        p_phi = np.random.uniform(0, np.pi)
        
        # Direction vector
        dx = np.cos(p_theta) * np.sin(p_phi)
        dy = np.sin(p_theta) * np.sin(p_phi)
        dz = np.cos(p_phi)
        
        # Add protrusion
        dist = (x*dx + y*dy + z*dz)
        mask = dist > 0.5
        protrusion = 0.5 * np.exp(-(1-dist[mask])**2 / 0.1)
        x[mask] += dx * protrusion
        y[mask] += dy * protrusion
        z[mask] += dz * protrusion
    
    # Add some noise
    if noise_level > 0:
        x += np.random.normal(0, noise_level * rx, n_vertices)
        y += np.random.normal(0, noise_level * ry, n_vertices)
        z += np.random.normal(0, noise_level * rz, n_vertices)
    
    # Combine points
    points = np.column_stack([x, y, z])
    
    # Generate faces using Delaunay triangulation
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    faces = tri.simplices
    
    # Convert to tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    faces_tensor = torch.tensor(faces, dtype=torch.int64)
    
    # Create random "features" for each point (e.g., curvature)
    features = torch.tensor(np.column_stack([
        np.random.normal(0, 0.1, n_vertices),  # Curvature
        np.random.normal(0, 0.1, n_vertices),  # Normal X
        np.random.normal(0, 0.1, n_vertices)   # Normal Y
    ]), dtype=torch.float32)
    
    return {
        'points': points_tensor.unsqueeze(0),  # Add batch dimension
        'features': features.unsqueeze(0),     # Add batch dimension
        'faces': faces_tensor.unsqueeze(0)     # Add batch dimension
    }

def visualize_mesh(mesh_data, output_path=None, show=True):
    """
    Visualize a 3D mesh from the model input format.
    
    Args:
        mesh_data: Dictionary with 'points' and 'faces' tensors
        output_path: Path to save the visualization
        show: Whether to show the plot
    """
    # Extract data
    points = mesh_data['points'][0].numpy()  # Remove batch dimension
    faces = mesh_data['faces'][0].numpy()    # Remove batch dimension
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh
    mesh = Poly3DCollection([points[face] for face in faces], alpha=0.5)
    mesh.set_edgecolor('k')
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)
    
    # Plot vertices
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='blue', alpha=0.5)
    
    # Set axis limits
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cell Mesh Visualization')
    
    # Save if requested
    if output_path:
        plt.savefig(output_path)
        print(f"Mesh visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def save_mesh_data(mesh_data, output_dir="sample_meshes"):
    """
    Save mesh data to file for later use.
    
    Args:
        mesh_data: Dictionary with 'points', 'features', and 'faces' tensors
        output_dir: Directory to save the data
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tensors
    torch.save(mesh_data, os.path.join(output_dir, "sample_mesh.pt"))
    print(f"Mesh data saved to {os.path.join(output_dir, 'sample_mesh.pt')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample 3D meshes for visualization")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--n_vertices", type=int, default=1000, help="Number of vertices in the mesh")
    parser.add_argument("--shape", type=str, default="cell", choices=["sphere", "cell"], help="Shape type")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level (0.0-1.0)")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots, just save them")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate the mesh
    if args.shape == "sphere":
        mesh_data = generate_sphere_mesh(n_vertices=args.n_vertices, noise_level=args.noise)
    else:
        mesh_data = generate_cell_like_mesh(n_vertices=args.n_vertices, noise_level=args.noise)
    
    # Visualize the mesh
    output_path = os.path.join(args.output_dir, f"{args.shape}_mesh.png") if args.output_dir else None
    visualize_mesh(mesh_data, output_path=output_path, show=not args.no_show)
    
    # Save mesh data for later use
    save_mesh_data(mesh_data, output_dir="sample_meshes") 