import os
import numpy as np
import networkx as nx
from typing import Dict, List


def build_context_features(embeddings: Dict[str, np.ndarray], rag_edges: List[tuple]) -> Dict[str, np.ndarray]:
    """
    Build MorphoContextFeatures by averaging each cell's embedding with its first-order neighbors.

    Args:
        embeddings: dict mapping sample_id -> 480-D numpy array
        rag_edges: list of tuples (u, v) representing edges in the region adjacency graph

    Returns:
        dict mapping sample_id -> 480-D context feature vector
    """
    # Build graph
    G = nx.Graph()
    G.add_edges_from(rag_edges)

    context_feats = {}
    for node, feat in embeddings.items():
        # Get neighbors
        neighbors = list(G.neighbors(node))
        if not neighbors:
            context_feats[node] = feat.copy()
            continue
        # Average own feature with neighbors
        neighbor_feats = [embeddings[n] for n in neighbors if n in embeddings]
        if neighbor_feats:
            context_feats[node] = np.mean([feat] + neighbor_feats, axis=0)
        else:
            context_feats[node] = feat.copy()
    return context_feats


def save_context_features(context_feats: Dict[str, np.ndarray], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for sid, vec in context_feats.items():
        np.save(os.path.join(out_dir, f"{sid}.npy"), vec) 