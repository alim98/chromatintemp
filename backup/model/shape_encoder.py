import torch
import torch.nn as nn
from torch_geometric.nn import DeepGCNLayer, GENConv
import torch_geometric.nn as gnn

class ShapeEncoder(nn.Module):
    """
    Shape encoder following MorphoFeatures paper specs:
    3 × GENConv blocks (DeepGCN) + global-max-pool + 2-layer MLP
    Output: 80-dimensional feature vector
    """
    def __init__(self, in_channels=6, hidden_channels=64, num_layers=3, output_dim=80):
        super(ShapeEncoder, self).__init__()
        
        # Initial convolution
        self.conv_in = GENConv(in_channels, hidden_channels)
        
        # 3 × GENConv blocks (DeepGCN)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DeepGCNLayer(
                    GENConv(hidden_channels, hidden_channels),
                    dropout=0.1,
                    act='relu'
                )
            )
        
        # Global max pooling
        self.pool = gnn.global_max_pool
        
        # 2-layer MLP for final embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Initial convolution
        x = self.conv_in(x, edge_index)
        x = torch.relu(x)
        
        # DeepGCN layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Global max pooling
        x = self.pool(x, batch)
        
        # MLP to get final embedding
        x = self.mlp(x)
        
        return x
