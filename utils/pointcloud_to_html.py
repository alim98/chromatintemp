
"""
Tool to convert point cloud PLY files to interactive 3D HTML files for visualization.
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import glob
import json
import shutil

def convert_pointcloud_to_html(input_file, output_path, title=None, point_size=0.05):
    """
    Convert a point cloud PLY file to an interactive 3D HTML file.
    
    Args:
        input_file (str): Path to the input PLY file
        output_path (str): Path to the output HTML file or directory
        title (str, optional): Title for the HTML page
        point_size (float, optional): Size of points in the visualization (default: 0.05)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        
        if os.path.isdir(output_path):
            
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_path, f"{base_name}.html")
        else:
            output_file = output_path
        
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        
        pcd = o3d.io.read_point_cloud(input_file)
        
        
        points = np.asarray(pcd.points)
        
        
        if not pcd.has_colors():
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            range_coords = max_coords - min_coords
            range_coords[range_coords == 0] = 1  
            
            
            colors = (points - min_coords) / range_coords
            colors = colors * 255  
        else:
            colors = np.asarray(pcd.colors) * 255  
        
        
        data = {
            'points': points.tolist(),
            'colors': colors.tolist()
        }
        
        
        if title is None:
            title = os.path.splitext(os.path.basename(input_file))[0]
        
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} - 3D Point Cloud Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        
        
        
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>{title}</h3>
        <p>Points: <span id="numPoints">0</span></p>
        <p>Controls: Left-click + drag to rotate, right-click + drag to pan, scroll to zoom</p>
    </div>
    <div id="controls">
        <label for="pointSize">Point Size:</label>
        <input type="range" id="pointSize" min="0.01" max="0.5" step="0.01" value="{point_size}">
        <span id="pointSizeValue">{point_size}</span>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Point cloud data
        const pointCloudData = {json.dumps(data)};
        
        // Three.js scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);
        
        // Camera
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;
        
        // Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        
        // Create point cloud
        const points = pointCloudData.points;
        const colors = pointCloudData.colors;
        
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colorValues = [];
        
        for (let i = 0; i < points.length; i++) {{
            positions.push(points[i][0], points[i][1], points[i][2]);
            colorValues.push(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255);
        }}
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorValues, 3));
        
        const material = new THREE.PointsMaterial({{
            size: {point_size},
            vertexColors: true
        }});
        
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);
        
        // Update point count
        document.getElementById('numPoints').textContent = points.length;
        
        // Point size slider
        const pointSizeSlider = document.getElementById('pointSize');
        const pointSizeValue = document.getElementById('pointSizeValue');
        
        pointSizeSlider.addEventListener('input', function() {{
            const size = parseFloat(this.value);
            material.size = size;
            pointSizeValue.textContent = size.toFixed(2);
        }});
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Center the point cloud
        const box = new THREE.Box3().setFromObject(pointCloud);
        const center = box.getCenter(new THREE.Vector3());
        pointCloud.position.sub(center);
        
        // Set camera position based on bounding box
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / Math.sin(fov / 2));
        camera.position.z = cameraZ * 1.5;
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        animate();
    </script>
</body>
</html>
"""
        
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Created HTML visualization at {output_file}")
        return True
    
    except Exception as e:
        print(f"Error converting {input_file} to HTML: {e}")
        return False

def process_directory(input_dir, output_dir, point_size=0.05):
    """
    Process all PLY files in a directory and convert them to HTML files.
    
    Args:
        input_dir (str): Directory containing PLY files
        output_dir (str): Directory to save HTML files
        point_size (float, optional): Size of points in the visualization (default: 0.05)
        
    Returns:
        int: Number of successfully converted files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    
    if not ply_files:
        print(f"No PLY files found in {input_dir}")
        return 0
    
    
    success_count = 0
    for ply_file in tqdm(ply_files, desc="Converting point clouds to HTML"):
        base_name = os.path.splitext(os.path.basename(ply_file))[0]
        html_file = os.path.join(output_dir, f"{base_name}.html")
        
        if convert_pointcloud_to_html(ply_file, html_file, title=f"Point Cloud: {base_name}", point_size=point_size):
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(ply_files)} files")
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Convert point cloud PLY files to interactive 3D HTML files")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input PLY file or directory containing PLY files")
    parser.add_argument("--output", type=str, required=True, 
                        help="Output HTML file or directory")
    parser.add_argument("--title", type=str, default=None, 
                        help="Title for the HTML page (only used for single file conversion)")
    parser.add_argument("--point_size", type=float, default=0.05,
                        help="Size of points in the visualization (default: 0.05)")
    
    args = parser.parse_args()
    
    
    if os.path.isdir(args.input):
        
        process_directory(args.input, args.output, point_size=args.point_size)
    else:
        
        if not args.input.endswith('.ply'):
            print(f"Error: Input file must be a PLY file: {args.input}")
            return
        
        
        convert_pointcloud_to_html(args.input, args.output, args.title, point_size=args.point_size)

if __name__ == "__main__":
    main() 