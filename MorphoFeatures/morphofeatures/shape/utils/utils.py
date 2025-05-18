import numpy as np
import networkx as nx


def mesh_to_graph(faces):
    """
    Convert a mesh represented by faces to a graph.
    
    Args:
        faces: Array of triangle faces (shape: N x 3)
        
    Returns:
        NetworkX graph representing the mesh connectivity
    """
    graph = nx.Graph()
    
    # Add all edges to the graph
    for face in faces:
        graph.add_edge(face[0], face[1])
        graph.add_edge(face[1], face[2])
        graph.add_edge(face[2], face[0])
    
    return graph


def read_off(file_path):
    """
    Read a mesh in OFF format.
    
    Args:
        file_path: Path to the OFF file
        
    Returns:
        Tuple of (vertices, faces) arrays
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Check if the first line contains OFF
    if "OFF" not in lines[0]:
        raise ValueError("Invalid OFF file format")
    
    # Parse the number of vertices and faces
    header = lines[1].strip().split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    
    # Parse vertices
    vertices = []
    for i in range(2, 2 + n_vertices):
        vertex = [float(x) for x in lines[i].strip().split()]
        vertices.append(vertex)
    
    # Parse faces
    faces = []
    for i in range(2 + n_vertices, 2 + n_vertices + n_faces):
        face = [int(x) for x in lines[i].strip().split()]
        # The first number in each line is the number of vertices per face
        faces.append(face)
    
    return np.array(vertices), np.array(faces)


def get_khop_neighbors(graph, node, k):
    """
    Get k-hop neighbors of a node in a graph.
    
    Args:
        graph: NetworkX graph
        node: Starting node
        k: Number of hops
        
    Returns:
        Array of node IDs that are within k hops of the starting node
    """
    # Get all nodes within k hops
    neighbors = set([node])  # Include the starting node
    frontier = set([node])
    
    for _ in range(k):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(graph.neighbors(n))
        frontier = next_frontier - neighbors  # Avoid revisiting nodes
        neighbors.update(frontier)
    
    return np.array(list(neighbors))
