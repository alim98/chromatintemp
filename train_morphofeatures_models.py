#!/usr/bin/env python
import os
import sys
import torch
import yaml
import argparse
import logging
import time
import traceback
from tqdm import tqdm
import numpy as np

# Configure detailed logging
logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s',
                   stream=sys.stdout,
                   level=logging.DEBUG)  # Change to DEBUG for more verbose output
logger = logging.getLogger(__name__)

try:
    # Add MorphoFeatures to the path
    sys.path.append(os.path.abspath("MorphoFeatures"))
    logger.info("Added MorphoFeatures to the path")

    # Import MorphoFeatures shape components
    try:
        from MorphoFeatures.morphofeatures.shape.train_shape_model import ShapeTrainer
        logger.info("Successfully imported ShapeTrainer")
    except ImportError as e:
        logger.error(f"Failed to import ShapeTrainer: {str(e)}")
        logger.error(traceback.format_exc())

    # Import custom dataloaders
    try:
        from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader
        logger.info("Successfully imported mesh dataloader")
    except ImportError as e:
        logger.error(f"Failed to import mesh dataloader: {str(e)}")
        logger.error(traceback.format_exc())
except Exception as e:
    logger.error(f"Setup error: {str(e)}")
    logger.error(traceback.format_exc())


class CustomShapeTrainer(ShapeTrainer):
    """
    A custom trainer that extends MorphoFeatures' ShapeTrainer to use our dataloaders.
    """
    def __init__(self, config):
        # Initialize the parent class
        logger.info("Initializing CustomShapeTrainer")
        super().__init__(config)
    
    def build_loaders(self):
        """
        Override the build_loaders method to use our custom dataloaders.
        """
        logger.info("Building custom shape dataloaders...")
        dataset_config = self.config['data']
        loader_config = self.config['loader']
        
        # Create train dataloader using our custom function
        self.train_loader = get_morphofeatures_mesh_dataloader(
            root_dir=dataset_config['root_dir'],
            class_csv_path=dataset_config.get('class_csv_path'),
            batch_size=loader_config.get('batch_size', 8),
            shuffle=loader_config.get('shuffle', True),
            num_workers=loader_config.get('num_workers', 0),  # Set to 0 to avoid multiprocessing issues
            precomputed_dir=dataset_config.get('precomputed_dir'),
            num_points=dataset_config.get('num_points', 1024),
            cache_dir=dataset_config.get('cache_dir'),
            debug=True  # Set to True for debugging
        )
        
        # Create validation dataloader
        if dataset_config.get('val_root_dir'):
            # If a separate validation directory is specified
            self.val_loader = get_morphofeatures_mesh_dataloader(
                root_dir=dataset_config['val_root_dir'],
                class_csv_path=dataset_config.get('val_class_csv_path', dataset_config.get('class_csv_path')),
                batch_size=loader_config.get('batch_size', 8),
                shuffle=False,
                num_workers=loader_config.get('num_workers', 0),  # Set to 0 to avoid multiprocessing issues
                precomputed_dir=dataset_config.get('precomputed_dir'),
                num_points=dataset_config.get('num_points', 1024),
                cache_dir=dataset_config.get('cache_dir'),
                debug=True  # Set to True for debugging
            )
        else:
            # Use the training data for validation (not ideal but works for small datasets)
            logger.warning("Using training data for validation. For proper evaluation, provide val_root_dir.")
            self.val_loader = self.train_loader


class TextureModelTrainer:
    """
    Trainer for texture models (both low-res and high-res)
    """
    def __init__(self, config, model_type='lowres'):
        logger.info(f"Initializing {model_type} TextureModelTrainer")
        try:
            # Using PyTorch's native libraries instead of neurofire
            self.config = config
            self.model_type = model_type
            self.device = torch.device(config.get('device', 'cpu'))
            logger.info(f"Using device: {self.device}")
            
            self.project_dir = config.get('project_directory', f'experiments/{model_type}_texture_model')
            os.makedirs(os.path.join(self.project_dir, 'Weights'), exist_ok=True)
            os.makedirs(os.path.join(self.project_dir, 'Logs'), exist_ok=True)
            logger.info(f"Created project directory: {self.project_dir}")
            
            # Try to import the custom dataloader adapter
            try:
                from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader
                self.get_morphofeatures_texture_dataloader = get_morphofeatures_texture_dataloader
                self.has_texture_dataloader = True
                logger.info("Successfully imported texture dataloader")
            except ImportError as e:
                logger.error(f"Could not import texture dataloader adapter: {e}")
                logger.error(traceback.format_exc())
                self.has_texture_dataloader = False
                
            # Setup the texture model
            self.setup_model()
            
            # Setup dataloaders
            self.setup_dataloaders()
        except Exception as e:
            logger.error(f"Error in TextureModelTrainer initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_model(self):
        """Set up the texture model, criterion, optimizer, and trainer"""
        logger.info("Setting up model")
        try:
            model_kwargs = self.config.get('model_kwargs', {})
            logger.debug(f"Model kwargs: {model_kwargs}")
            
            # Create a custom UNet3D model instead of using neurofire
            self.model = UNet3D(**model_kwargs).to(self.device)
            logger.info(f"Created UNet3D model: {self.model}")
            
            # Compile the criterion
            criterion_name = self.config.get('loss', 'MSELoss')
            criterion_kwargs = self.config.get('loss_kwargs', {})
            logger.debug(f"Criterion: {criterion_name}, kwargs: {criterion_kwargs}")
            
            if isinstance(criterion_name, str):
                self.criterion = getattr(torch.nn, criterion_name)(**criterion_kwargs)
            else:
                self.criterion = None
            logger.info(f"Created criterion: {self.criterion}")
            
            # Build optimizer
            optimizer_config = self.config.get('training_optimizer_kwargs', {})
            optimizer_method = optimizer_config.pop('optimizer', 'Adam')
            optimizer_kwargs = optimizer_config.pop('optimizer_kwargs', {}) if 'optimizer_kwargs' in optimizer_config else {}
            logger.debug(f"Optimizer: {optimizer_method}, kwargs: {optimizer_kwargs}")
            
            self.optimizer = getattr(torch.optim, optimizer_method)(
                self.model.parameters(), 
                **optimizer_kwargs
            )
            logger.info(f"Created optimizer: {self.optimizer}")
            
            # Setup scheduler for learning rate
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=0.98,
                patience=100,
                verbose=True
            )
            logger.info("Created learning rate scheduler")
            
            # Best validation score tracking
            self.best_val_loss = float('inf')
            self.val_loss_momentum = 0
            self.smoothness = self.config.get('smoothness', 0.95)
            logger.info("Model setup complete")
        except Exception as e:
            logger.error(f"Error in setup_model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def setup_dataloaders(self):
        """Setup custom texture dataloaders for training and validation"""
        logger.info(f"Setting up {self.model_type} texture dataloaders")
        try:
            # Get dataset config
            data_config = self.config.get('data_config', {})
            # Override root_dir with the absolute path to the data directory
            root_dir = '/teamspace/studios/this_studio'
            logger.info(f"Using absolute path for root_dir: {root_dir}")
            
            class_csv_path = data_config.get('class_csv_path', None)
            logger.debug(f"Data config - root_dir: {root_dir}, class_csv_path: {class_csv_path}")
            
            # Get specific config for texture data
            is_cytoplasm = data_config.get('is_cytoplasm', False)
            box_size = data_config.get('box_size', (104, 104, 104))
            split = data_config.get('split', 0.2)
            seed = data_config.get('seed', 42)
            logger.debug(f"Data params - is_cytoplasm: {is_cytoplasm}, box_size: {box_size}, split: {split}, seed: {seed}")
            
            # Get loader configs
            train_loader_config = self.config.get('loader_config', {})
            val_loader_config = self.config.get('val_loader_config', {})
            logger.debug(f"Loader configs - train: {train_loader_config}, val: {val_loader_config}")
            
            # Check if custom dataloader is available
            if not hasattr(self, 'has_texture_dataloader') or not self.has_texture_dataloader:
                logger.error("Custom texture dataloader is not available")
                raise ImportError("Custom texture dataloader is not available. Install required dependencies.")
            
            # Check if root directory exists
            if not os.path.exists(root_dir):
                logger.error(f"Root directory does not exist: {root_dir}")
                raise FileNotFoundError(f"Root directory not found: {root_dir}")
            
            # Check if data directory contains any samples
            samples_in_dir = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            logger.info(f"Found {len(samples_in_dir)} sample directories in root_dir")
            
            # Check if the nuclei_sample_1a_v1 directory exists
            sample_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
            if not os.path.exists(sample_dir):
                logger.error(f"Sample directory does not exist: {sample_dir}")
                raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
            
            # Check if required files exist
            raw_path = os.path.join(sample_dir, 'raw.npy')
            label_path = os.path.join(sample_dir, 'texture_label.npy')
            
            logger.info(f"Checking for raw.npy: {os.path.exists(raw_path)}")
            logger.info(f"Checking for texture_label.npy: {os.path.exists(label_path)}")
            
            if not os.path.exists(raw_path) or not os.path.exists(label_path):
                logger.warning(f"Required files missing in {sample_dir}")
                logger.warning(f"Available files: {os.listdir(sample_dir)}")
                logger.warning("Will generate dummy data for missing files")
                
                # Create the raw.npy and texture_label.npy files for testing
                if not os.path.exists(raw_path):
                    logger.info("Creating dummy raw.npy file for testing")
                    raw_vol = np.random.rand(200, 200, 200).astype(np.float32)
                    np.save(raw_path, raw_vol)
                
                if not os.path.exists(label_path):
                    logger.info("Creating dummy texture_label.npy file for testing")
                    label_vol = np.random.rand(200, 200, 200).astype(np.float32) > 0.5
                    np.save(label_path, label_vol)
            
            # Use the absolute path to the sample directory
            sample_path = sample_dir
            
            # Use the full path for both training and validation
            train_ids = [sample_path]
            val_ids = [sample_path]
            logger.info(f"Using sample for both training and validation: {sample_path}")
            
            # Create the dataloaders with full paths to samples
            logger.info(f"Creating training dataloader with {len(train_ids)} samples")
            self.train_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,  # This is not used when full paths are provided
                sample_ids=train_ids,  # Now using full paths
                batch_size=train_loader_config.get('batch_size', 4),
                shuffle=train_loader_config.get('shuffle', True),
                num_workers=train_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=box_size
            )
            logger.info(f"Created training dataloader with {len(self.train_loader.dataset)} samples")
            
            logger.info(f"Creating validation dataloader with {len(val_ids)} samples")
            self.val_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,  # This is not used when full paths are provided
                sample_ids=val_ids,  # Now using full paths
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=val_loader_config.get('shuffle', False),
                num_workers=val_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=box_size
            )
            logger.info(f"Created validation dataloader with {len(self.val_loader.dataset)} samples")
        except Exception as e:
            logger.error(f"Error in setup_dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def train(self):
        """Train the model"""
        logger.info("Starting training")
        try:
            num_epochs = self.config.get('num_epochs', 50)
            logger.info(f"Training for {num_epochs} epochs")
            
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                
                # Training loop
                self.model.train()
                train_loss = 0.0
                
                for i, batch in enumerate(tqdm(self.train_loader, desc=f"Training epoch {epoch+1}")):
                    inputs, targets = batch
                    logger.debug(f"Batch {i+1} - input shape: {inputs.shape}, target shape: {targets.shape}")
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # Validate every 100 iterations
                    if (i+1) % 100 == 0:
                        logger.info(f"Validating at iteration {i+1}")
                        self.validate()
                
                # Calculate average training loss
                avg_train_loss = train_loss / len(self.train_loader)
                logger.info(f"Training Loss: {avg_train_loss:.4f}")
                
                # Validate at the end of each epoch
                val_loss = self.validate()
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save model if validation loss improved
                if val_loss < self.best_val_loss:
                    logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                    self.best_val_loss = val_loss
                    
                    # Save model
                    model_path = os.path.join(self.project_dir, 'Weights', f'best_model_epoch_{epoch+1}.pt')
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def validate(self):
        """Validate the model"""
        logger.info("Validating model")
        try:
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                    inputs, targets = batch
                    logger.debug(f"Validation batch {i+1} - input shape: {inputs.shape}, target shape: {targets.shape}")
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Update the momentum-based validation loss for learning rate scheduling
            if self.val_loss_momentum == 0:
                self.val_loss_momentum = avg_val_loss
            else:
                self.val_loss_momentum = self.smoothness * self.val_loss_momentum + (1 - self.smoothness) * avg_val_loss
            
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            return avg_val_loss
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Custom 3D UNet implementation using PyTorch
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x

class UpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class UNet3D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f_maps=[32, 64, 128, 256], final_sigmoid=True):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_maps = f_maps
        self.final_sigmoid = final_sigmoid
        
        # Initial convolution block
        self.inc = ConvBlock(in_channels, f_maps[0])
        
        # Down path
        self.down1 = DownBlock(f_maps[0], f_maps[1])
        self.down2 = DownBlock(f_maps[1], f_maps[2])
        self.down3 = DownBlock(f_maps[2], f_maps[3])
        
        # Up path
        self.up1 = UpBlock(f_maps[3], f_maps[2])
        self.up2 = UpBlock(f_maps[2], f_maps[1])
        self.up3 = UpBlock(f_maps[1], f_maps[0])
        
        # Final convolution
        self.outc = torch.nn.Conv3d(f_maps[0], out_channels, kernel_size=1)
        
        # Final activation
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # Down path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Up path
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Final convolution
        x = self.outc(x)
        
        # Apply sigmoid if required
        if self.final_sigmoid:
            x = self.sigmoid(x)
        
        return x

def train_shape_model(config_path):
    """Train the shape model using the given configuration"""
    logger.info(f"Training shape model with config: {config_path}")
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config: {config}")
        
        # Create the trainer
        trainer = CustomShapeTrainer(config)
        
        # Train the model
        trainer.fit()
    except Exception as e:
        logger.error(f"Error in train_shape_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_texture_model(config_path, model_type='lowres'):
    """Train the texture model using the given configuration"""
    logger.info(f"Training {model_type} texture model with config: {config_path}")
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.debug(f"Loaded config: {config}")
        
        # Create the trainer
        trainer = TextureModelTrainer(config, model_type)
        
        # Ensure experiment directory exists
        project_dir = config.get('project_directory', f'experiments/{model_type}_texture_model')
        os.makedirs(project_dir, exist_ok=True)
        
        # Save the configuration to the experiment directory
        with open(os.path.join(project_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        
        # Train the model
        trainer.train()
    except Exception as e:
        logger.error(f"Error in train_texture_model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main entry point"""
    try:
        logger.info("Starting MorphoFeatures model training")
        
        parser = argparse.ArgumentParser(description='Train MorphoFeatures models')
        parser.add_argument('--model', type=str, required=True, choices=['shape', 'lowres', 'highres'],
                            help='Type of model to train: shape, lowres (texture), or highres (texture)')
        parser.add_argument('--config', type=str, required=True,
                            help='Path to the configuration file')
        
        args = parser.parse_args()
        logger.info(f"Parsed arguments: model={args.model}, config={args.config}")
        
        if args.model == 'shape':
            train_shape_model(args.config)
        elif args.model == 'lowres':
            train_texture_model(args.config, model_type='lowres')
        elif args.model == 'highres':
            train_texture_model(args.config, model_type='highres')
        else:
            logger.error(f"Unknown model type: {args.model}")
            raise ValueError(f"Unknown model type: {args.model}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Uncaught exception: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 