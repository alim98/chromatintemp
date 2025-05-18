import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tiff_dataloader import get_tiff_dataloader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class UNet3D(nn.Module):
    """A simplified 3D UNet model."""
    def __init__(self, in_channels=1, out_channels=1, f_maps=[32, 64, 128, 256], final_sigmoid=True):
        super(UNet3D, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._make_encoder_block(in_channels, f_maps[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_encoder_block(f_maps[0], f_maps[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = self._make_encoder_block(f_maps[1], f_maps[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = self._make_encoder_block(f_maps[2], f_maps[3])
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose3d(f_maps[3], f_maps[2], kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(f_maps[3], f_maps[2])
        
        self.upconv2 = nn.ConvTranspose3d(f_maps[2], f_maps[1], kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(f_maps[2], f_maps[1])
        
        self.upconv1 = nn.ConvTranspose3d(f_maps[1], f_maps[0], kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(f_maps[1], f_maps[0])
        
        # Output layer
        self.outconv = nn.Conv3d(f_maps[0], out_channels, kernel_size=1)
        
        # Final activation
        self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Identity()
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bridge
        bridge = self.bridge(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.dec3(torch.cat([self.upconv3(bridge), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        # Output
        out = self.outconv(dec1)
        return self.final_activation(out)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config):
    """Train the texture model using the provided configuration."""
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create project directory
    os.makedirs(config['project_directory'], exist_ok=True)
    logger.info(f"Created project directory: {config['project_directory']}")
    
    # Setup dataloaders
    logger.info("Setting up dataloaders")
    data_config = config['data_config']
    loader_config = config['loader_config']
    val_loader_config = config['val_loader_config']
    
    # Get target size from config (box_size)
    target_size = tuple(data_config.get('box_size', (104, 104, 104)))
    logger.info(f"Using target volume size: {target_size}")
    
    # Create a subset of the samples for faster training
    sample_limit = 10
    logger.info(f"Using only {sample_limit} samples for training (to speed up the process)")
    
    train_loader = get_tiff_dataloader(
        root_dir=data_config['root_dir'],
        batch_size=loader_config['batch_size'],
        shuffle=loader_config['shuffle'],
        num_workers=loader_config['num_workers'],
        input_dir=data_config.get('input_dir', 'raw'),
        target_dir=data_config.get('target_dir', 'mask'),
        target_size=target_size
    )
    
    val_loader = get_tiff_dataloader(
        root_dir=data_config['root_dir'],
        batch_size=val_loader_config['batch_size'],
        shuffle=val_loader_config['shuffle'],
        num_workers=val_loader_config['num_workers'],
        input_dir=data_config.get('input_dir', 'raw'),
        target_dir=data_config.get('target_dir', 'mask'),
        target_size=target_size
    )
    
    # Setup model
    logger.info("Setting up model")
    model_kwargs = config['model_kwargs']
    model = UNet3D(**model_kwargs).to(device)
    logger.info(f"Created UNet3D model")
    
    # Setup loss function
    loss_name = config['loss']
    loss_kwargs = config['loss_kwargs']
    if loss_name == 'BCELoss':
        criterion = nn.BCELoss(**loss_kwargs)
    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    logger.info(f"Created criterion: {loss_name}")
    
    # Setup optimizer
    optimizer_config = config['training_optimizer_kwargs']
    optimizer_name = optimizer_config['optimizer']
    optimizer_kwargs = optimizer_config['optimizer_kwargs']
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    logger.info(f"Created optimizer: {optimizer_name}")
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    logger.info("Created learning rate scheduler")
    
    # Training loop
    num_epochs = config.get('num_epochs_debug', 3)  # Use fewer epochs for debugging
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Limit to a few batches for faster debugging
            if batch_idx >= 5:  
                logger.info(f"Limiting to first {batch_idx+1} batches for faster debugging")
                break
        
        avg_train_loss = train_loss / (min(len(train_loader), 6))
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Limit validation to a few batches
                if batch_idx >= 2:
                    break
        
        avg_val_loss = val_loss / (min(len(val_loader), 3))
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(config['project_directory'], f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")
    
    logger.info("Training completed")


def main():
    parser = argparse.ArgumentParser(description="Train a 3D UNet model on TIFF volume data")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    logger.info("Starting model training")
    train_model(config)


if __name__ == "__main__":
    main() 