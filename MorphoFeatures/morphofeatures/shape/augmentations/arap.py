import argparse
import random
import numpy as np
import igl
import vedo
import time

from MorphoFeatures.morphofeatures.shape.utils import mesh_to_graph, read_off, get_khop_neighbors
# the repo doesnt have the shape.utils.py file, so i created it myself

class AsRigidAsPossibleDeformation:

    def __init__(self,
                 region_size,
                 num_regions,
                 deform_scale_low=0.0005,
                 deform_scale_high=0.025,
                 deformation='random',
                 smoothing='random'):
        """
        Args:
          region_size: Upper bound on the region size in khops (int)
          num_regions: Upper bound on the number of regions to deform (int)
          deformation: determines in which direction a region
            is deformed. (str) (options: {'random', 'constant', 'normal'})
          smoothing: if smoothing should be applied in the form of
            biharmonic coordinates. (options: {'random', True, False})
        """
        self.region_size = region_size
        self.num_regions = num_regions
        self.scale_low = deform_scale_low
        self.scale_high = deform_scale_high
        self.deformation = deformation
        self.smoothing = smoothing

    def deform(self, v, f, b, mode, selection):
        bc = np.zeros((b.size, v.shape[1]))
        if mode == 'constant':
            deform_vecs = {}
            for idx in np.unique(selection):
                deform_vec = random.uniform(self.scale_low, self.scale_high) * np.random.randn(3)
                deform_vecs[int(idx)] = deform_vec if idx != -1 else np.zeros(3)
            for i in range(0, b.size):
                bc[i] = v[b[i]] + deform_vecs[int(selection[b[i]])]
        elif mode == 'normal':
            normals = igl.per_vertex_normals(v, f)
            factors = {
                int(idx): random.choice([-1., 1.]) * random.uniform(self.scale_low, self.scale_high)
                for idx in np.unique(selection)
            }
            factors[0] = 0
            for i in range(0, b.size):
                bc[i] = v[b[i]] + factors[int(selection[b[i]])] * normals[b[i]]
        return bc

    def smooth(self, v, f, vn, handle_dict):
        ctrl_ids = random.sample(
            list(range(v.shape[0])), int(v.shape[0] / 3)
        )
        for node_id, ids in handle_dict.items():
            ctrl_ids = np.setdiff1d(
                np.asarray(ctrl_ids), ids['smoothing_ids']
            ).tolist()
        b = np.array(ctrl_ids)
        bw = igl.biharmonic_coordinates(vn, f, [[i] for i in ctrl_ids], k=2)
        vn = (bw @ vn[b[:]]) 
        return vn

    def __call__(self, v, f):
        graph = mesh_to_graph(f)
        node_ids = graph.nodes().tolist() 
        selection = -1. * np.ones(v.shape[0])

        handle_dict = {}
        num_regions = random.randint(1, self.num_regions)
        for i in range(num_regions):
            if not node_ids:  # if node_ids is an empty list
                break
            node_id = random.sample(node_ids, 1)[0]
            inner_sz = random.randint(1, max(1, self.region_size - 1))
            outer_sz = inner_sz + self.region_size
            inner_ids = get_khop_neighbors(
                graph, node_id, k=inner_sz
            )
            outer_ids = get_khop_neighbors(
                graph, node_id, k=outer_sz
            )
            node_ids = list(set(node_ids) - set(outer_ids.tolist()))
            smoothing_ids = np.setdiff1d(
                get_khop_neighbors(graph, node_id, k=inner_sz + min(3, inner_sz)),
                inner_ids
            )
            selection[inner_ids] = float(i)
            handle_dict[node_id] = {'smoothing_ids': smoothing_ids}

        # Vertices in selection
        b = np.array(
            [[t[0] for t in [(i, selection[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]
        ).T
        
        arap = igl.ARAP(v, f, 3, b)
        
        # Select deformation mode
        if self.deformation == 'random':
            deformation = 'constant' if random.random() > .5 else 'normal'
        else:
            deformation = self.deformation

        bc = self.deform(v, f, b, deformation, selection)
        vn = arap.solve(bc, v)

        # Select smoothing mode
        if self.smoothing == 'random':
            smoothing = True if random.random() > .5 else False
        else:
            smoothing = self.smoothing

        if smoothing:
            vn = self.smooth(v, f, vn, handle_dict)
        return vn, selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_mesh', type=str, required=True)
    parser.add_argument('--region_size', type=int, default=10)
    parser.add_argument('--num_regions', type=int, default=9)
    parser.add_argument('--deform_scale_low', type=float, default=0.005)
    parser.add_argument('--deform_scale_high', type=float, default=0.05)
    parser.add_argument('--deformation', default='random')
    parser.add_argument('--smoothing', type=bool, default=True)
    args = parser.parse_args()

    v, f = read_off(args.path_to_mesh)
    v, f = np.asarray(v), np.asarray(f)[:, 1:]
    transform = AsRigidAsPossibleDeformation(
        region_size=args.region_size,
        num_regions=args.num_regions,
        deform_scale_low=args.deform_scale_low,
        deform_scale_high=args.deform_scale_high,
        deformation=args.deformation,
        smoothing=args.smoothing
    )
    t0 = time.time()
    vn, selection = transform(v, f)
    print('time', time.time() - t0)
    plt = vedo.Plotter(N=2, axes=0)
    mesh = vedo.Mesh([v, f])
    mesh.cmap('rainbow', selection) 
    mesh.addScalarBar3D()
    plt.show(mesh, at=0, interactive=0)
    deformed_mesh = vedo.Mesh([vn, f])
    deformed_mesh.cmap('rainbow', selection) 
    deformed_mesh.addScalarBar3D()
    plt.show(deformed_mesh, at=1, interactive=1)

def arap_warp(points, region_size=5, num_regions=3, 
              deform_scale_low=0.001, deform_scale_high=0.02,
              deformation='random', smoothing='random'):
    """
    Apply a simplified warping to a point cloud that mimics aspects of ARAP deformation.
    Since full ARAP requires mesh connectivity, this function applies localized deformations
    to simulate the effect.
    
    Args:
        points (numpy.ndarray): Point cloud data of shape (N, 3)
        region_size (int): Upper bound on the region size in khops (used to determine region radius)
        num_regions (int): Upper bound on the number of regions to deform
        deform_scale_low (float): Lower bound of deformation scale
        deform_scale_high (float): Upper bound of deformation scale
        deformation (str): Deformation strategy ('random', 'constant', or 'normal')
        smoothing (str or bool): Whether to apply smoothing (ignored in this implementation)
        
    Returns:
        numpy.ndarray: Deformed point cloud
    """
    try:
        # Center points for better processing
        points_centered = points - np.mean(points, axis=0, keepdims=True)
        
        # Determine scale for normalization
        scale = np.max(np.abs(points_centered))
        points_normalized = points_centered / scale
        
        # Create a copy for modification
        deformed_points = points_normalized.copy()
        
        # Calculate bounding box
        min_coords = np.min(points_normalized, axis=0)
        max_coords = np.max(points_normalized, axis=0)
        bbox_size = max_coords - min_coords
        
        # Determine number of regions to deform
        num_actual_regions = np.random.randint(1, num_regions + 1)
        
        # For each region, apply a localized deformation
        for _ in range(num_actual_regions):
            # Choose a random center for this region
            center = min_coords + np.random.random(3) * bbox_size
            
            # Determine region radius based on region_size parameter (normalized)
            radius = (region_size / 20.0) * np.mean(bbox_size)
            
            # Calculate distances from all points to this center
            distances = np.sqrt(np.sum((points_normalized - center)**2, axis=1))
            
            # Create a smooth falloff for the deformation (1 at center, 0 at radius)
            weights = np.maximum(0, 1.0 - distances / radius)
            
            # If no points in this region, skip
            if np.max(weights) < 1e-5:
                continue
                
            # Generate a random deformation vector
            if deformation == 'random' or deformation == 'constant':
                # Random direction with random magnitude
                deform_vec = np.random.randn(3)
                deform_vec = deform_vec / np.linalg.norm(deform_vec)
                magnitude = np.random.uniform(deform_scale_low, deform_scale_high)
                deform_vec *= magnitude
            else:  # 'normal' or anything else
                # Use point normals if we can approximate them (gradient of distance field)
                # For simplicity, just use direction from center
                deform_vec = np.zeros((points_normalized.shape[0], 3))
                for i in range(points_normalized.shape[0]):
                    if weights[i] > 1e-5:
                        # Direction from center to point
                        direction = points_normalized[i] - center
                        if np.linalg.norm(direction) > 1e-5:
                            direction = direction / np.linalg.norm(direction)
                        else:
                            direction = np.random.randn(3)
                            direction = direction / np.linalg.norm(direction)
                        
                        # Scale by weight and random magnitude
                        magnitude = np.random.uniform(deform_scale_low, deform_scale_high)
                        deform_vec[i] = direction * magnitude * weights[i]
            
            # Apply the deformation with weights as multipliers
            if deformation == 'random' or deformation == 'constant':
                # Apply same deformation vector to all points, scaled by weight
                for i in range(points_normalized.shape[0]):
                    deformed_points[i] += deform_vec * weights[i]
            else:
                # Apply unique deformation to each point
                deformed_points += deform_vec
        
        # Rescale back to original scale and add original mean
        deformed_points = deformed_points * scale + np.mean(points, axis=0, keepdims=True)
        
        return deformed_points
        
    except Exception as e:
        # If anything fails, return original points
        print(f"ARAP warping failed: {e}")
        return points
