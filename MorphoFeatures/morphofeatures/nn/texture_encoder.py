import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    """Downsampling block with maxpool followed by convolution"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x

class TextureEncoder(nn.Module):
    """
    3D UNet encoder for texture feature extraction as described in the paper:
    - 3-block architecture
    - Feature maps: 64→128→256
    - Global average pooling
    - Output: 80-D features
    """
    def __init__(self, in_channels=1, out_channels=80, f_maps=[64, 128, 256], dropout=0.0):
        super(TextureEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_maps = f_maps
        
        # Initial convolution block
        self.inc = ConvBlock(in_channels, f_maps[0])
        
        # Down path with 3 blocks as specified in the paper
        self.down1 = DownBlock(f_maps[0], f_maps[1])
        self.down2 = DownBlock(f_maps[1], f_maps[2])
        
        # Final FC layers to convert to 80-D embeddings
        self.fc = nn.Sequential(
            nn.Linear(f_maps[2], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels)
        )
        
        # Optional projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        
        # UNet decoder for autoencoder branch (used in texture models)
        self.up1 = nn.ConvTranspose3d(f_maps[2], f_maps[1], kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(f_maps[2], f_maps[1])
        
        self.up2 = nn.ConvTranspose3d(f_maps[1], f_maps[0], kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(f_maps[1], f_maps[0])
        
        self.out_conv = nn.Conv3d(f_maps[0], in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        """Encoder part only - returns features before FC layer and embeddings"""
        # Down path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Global average pooling
        pooled = F.adaptive_avg_pool3d(x3, 1).view(x3.size(0), -1)
        
        # FC to get embeddings
        embeddings = self.fc(pooled)
        
        return x1, x2, x3, embeddings
    
    def decode(self, x1, x2, x3):
        """Decoder part for autoencoder"""
        # Up path
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv2(x)
        
        # Output layer
        x = self.out_conv(x)
        x = self.sigmoid(x)
        
        return x
    
    def forward(self, x):
        """Full forward pass - returns embedding & reconstructed volume"""
        # Encode
        x1, x2, x3, embeddings = self.encode(x)
        
        # Return different outputs based on training mode
        if self.training and hasattr(self, 'up1'):
            # During training with autoencoder loss, also return reconstruction
            reconstructed = self.decode(x1, x2, x3)
            projection = self.projection_head(embeddings)
            return projection, embeddings, reconstructed
        else:
            # During inference or contrastive-only training, just return embeddings
            projection = self.projection_head(embeddings)
            return projection, embeddings

    def transfer_weights(self, state_dict):
        """
        Transfer weights from a checkpoint
        Useful when transferring between cyto/nucleus models
        """
        # Filter out projection head and decoder weights
        encoder_dict = {k: v for k, v in state_dict.items() 
                        if not k.startswith('projection_head') and 
                           not any(x in k for x in ['up1', 'up2', 'up_conv', 'out_conv'])}
        
        # Load encoder weights
        model_dict = self.state_dict()
        model_dict.update(encoder_dict)
        self.load_state_dict(model_dict)


if __name__ == "__main__":
    # Quick test of the model
    model = TextureEncoder()
    
    # Test encoder with a small 3D volume
    x = torch.randn(2, 1, 32, 32, 32)
    proj, embeddings, reconstructed = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")  # Should be [2, 80]
    print(f"Projection shape: {proj.shape}")       # Should be [2, 64]
    print(f"Reconstructed shape: {reconstructed.shape}")  # Should match input 