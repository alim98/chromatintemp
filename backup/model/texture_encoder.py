import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureEncoder(nn.Module):
    """
    Texture encoder following MorphoFeatures paper specs:
    3-block 3-D UNet encoder (64→128→256 fm) + Global Average Pooling
    Output: 80-dimensional feature vector
    Weights shared between cytoplasm & nucleus
    """
    def __init__(self, in_channels=1, output_dim=80):
        super(TextureEncoder, self).__init__()
        
        # Encoder pathway with exact feature maps as per paper
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Pooling operations
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Final linear layer to get to 80-D
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # Encoder pathway
        x1 = self.enc1(x)
        x = self.pool(x1)
        
        x2 = self.enc2(x)
        x = self.pool(x2)
        
        x3 = self.enc3(x)
        
        # Global Average Pooling
        x = self.gap(x3)
        x = x.view(x.size(0), -1)
        
        # Project to 80-D
        x = self.fc(x)
        
        return x
