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
import pandas as pd

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
    
    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Training epoch {self.epoch}')):
            # The batch might be a dictionary or a tuple depending on the dataloader
            # Let's examine the batch to determine its format
            if isinstance(batch, dict):
                # Original format: dictionary with 'points' and 'features'
                points = batch['points'].to(self.device)
                features = batch['features'].to(self.device) if 'features' in batch else None
            else:
                # Our custom format: tuple of (data, labels)
                points = batch[0].to(self.device)
                features = None  # Features are integrated in the points data
            
            # Forward pass (match the original implementation's signature)
            if features is not None:
                out, h = self.model(points, features)
            else:
                out, h = self.model(points)
            
            # Create labels for contrastive learning
            # Make sure we have enough labels for the embeddings
            embedding_count = out.size(0)
            if embedding_count % 2 == 0:
                # If even number of embeddings, create pairs
                labels = torch.arange(embedding_count // 2).repeat_interleave(2).to(self.device)
            else:
                # If odd number of embeddings, adjust to make sure labels match embeddings
                # Here we'll treat each embedding as its own class for simplicity
                labels = torch.arange(embedding_count).to(self.device)
                
            logger.debug(f"Embeddings shape: {out.shape}, Labels shape: {labels.shape}")
            
            # Calculate loss
            loss = self.criterion(out, labels)
            
            # Skip iteration if loss is NaN
            if torch.isnan(loss).item():
                logger.warning(f'NaN loss encountered: {loss.item()}')
                continue
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
            
            # Update global step counter
            self.step += 1
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {self.epoch} - Average training loss: {avg_loss:.6f}")
        return avg_loss
    
    def validate_epoch(self):
        """Validate the model"""
        if self.epoch % self.config['training']['validate_every'] != 0:
            return
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                # The batch might be a dictionary or a tuple depending on the dataloader
                # Let's examine the batch to determine its format
                if isinstance(batch, dict):
                    # Original format: dictionary with 'points' and 'features'
                    points = batch['points'].to(self.device)
                    features = batch['features'].to(self.device) if 'features' in batch else None
                else:
                    # Our custom format: tuple of (data, labels)
                    points = batch[0].to(self.device)
                    features = None  # Features are integrated in the points data
                
                # Forward pass (match the original implementation's signature)
                if features is not None:
                    out, h = self.model(points, features)
                else:
                    out, h = self.model(points)
                
                # Create labels for contrastive learning
                # Make sure we have enough labels for the embeddings
                embedding_count = out.size(0)
                if embedding_count % 2 == 0:
                    # If even number of embeddings, create pairs
                    labels = torch.arange(embedding_count // 2).repeat_interleave(2).to(self.device)
                else:
                    # If odd number of embeddings, adjust to make sure labels match embeddings
                    # Here we'll treat each embedding as its own class for simplicity
                    labels = torch.arange(embedding_count).to(self.device)
                    
                logger.debug(f"Validation - Embeddings shape: {out.shape}, Labels shape: {labels.shape}")
                
                # Calculate loss
                loss = self.criterion(out, labels)
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {self.epoch} - Validation loss: {avg_loss:.6f}")
        
        # Save best model
        if self.best_val_loss is None or avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.checkpoint(True)
            logger.info(f"New best validation loss: {avg_loss:.6f}")
            
        return avg_loss
    
    def train(self):
        """Train the model for all epochs"""
        for epoch_num in tqdm(range(self.config['training']['epochs']), desc='Epochs'):
            self.epoch = epoch_num
            self.train_epoch()
            self.validate_epoch()
            self.scheduler.step()
            self.checkpoint(False)
            
    def run(self):
        """Main training function (called by fit)"""
        # Check if wandb should be used
        use_wandb = self.config.get('use_wandb', False)
        self.use_wandb = use_wandb
        
        if use_wandb:
            try:
                import wandb
                with wandb.init(project=self.config.get('wandb_project', 'MorphoFeatures')):
                    self.validate_epoch()
                    self.train()
            except (ImportError, AttributeError) as e:
                logger.warning(f"Unable to use wandb for logging: {e}")
                logger.warning("Continuing without wandb logging...")
                self.use_wandb = False
                self.validate_epoch()
                self.train()
        else:
            logger.info("Wandb logging disabled in config. Using console output only.")
            self.validate_epoch()
            self.train()
            
    def fit(self):
        """Wrapper method to match the expected API in train_morphofeatures_models.py"""
        logger.info("Starting model training via fit()...")
        self.run()


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
            
            # Get parameters from config
            use_tiff = data_config.get('use_tiff', False)
            root_dir = data_config.get('root_dir', '/teamspace/studios/this_studio')
            class_csv_path = data_config.get('class_csv_path', None)
            is_cytoplasm = data_config.get('is_cytoplasm', False)
            box_size = data_config.get('box_size', (104, 104, 104))
            input_dir = data_config.get('input_dir', 'raw')
            target_dir = data_config.get('target_dir', 'mask')
            
            # Get loader configs
            train_loader_config = self.config.get('loader_config', {})
            val_loader_config = self.config.get('val_loader_config', {})
            
            logger.info(f"Reading CSV: {class_csv_path}")
            
            # Directly read the CSV file
            sample_ids = []
            if class_csv_path and os.path.exists(class_csv_path):
                # Specify dtype={'sample_id': str} to preserve leading zeros
                df = pd.read_csv(class_csv_path, dtype={'sample_id': str})
                logger.info(f"CSV contains {len(df)} samples with columns: {df.columns.tolist()}")
                
                if 'sample_id' in df.columns:
                    # Extract sample IDs and convert to strings
                    csv_samples = [str(id) for id in df['sample_id'].unique()]
                    logger.info(f"Found {len(csv_samples)} unique sample IDs in CSV: {csv_samples}")
                    
                    # Create full paths for sample directories
                    # Check if root_dir is the nuclei_sample_1a_v1 directory or its parent
                    if os.path.basename(root_dir) == 'nuclei_sample_1a_v1':
                        parent_dir = root_dir
                    else:
                        parent_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
                    
                    # Log the parent directory we're using
                    logger.info(f"Using parent directory for samples: {parent_dir}")
                    
                    for sample_id in csv_samples:
                        sample_path = os.path.join(parent_dir, sample_id)
                        logger.info(f"Checking sample path: {sample_path}")
                        if os.path.exists(sample_path) and os.path.isdir(sample_path):
                            # Verify that this directory contains 'raw' and 'mask' folders
                            raw_dir = os.path.join(sample_path, input_dir)
                            mask_dir = os.path.join(sample_path, target_dir)
                            if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                                logger.info(f"Found valid sample with raw/mask dirs: {sample_path}")
                                sample_ids.append(sample_path)
                            else:
                                logger.warning(f"Sample directory doesn't have required directories: {sample_path}")
                                logger.warning(f"  Missing: {input_dir if not os.path.exists(raw_dir) else ''} {target_dir if not os.path.exists(mask_dir) else ''}")
                        else:
                            logger.warning(f"Sample directory not found: {sample_path}")
                else:
                    logger.error(f"CSV does not contain 'sample_id' column")
            else:
                logger.warning(f"CSV file not found: {class_csv_path}")
            
            if not sample_ids:
                logger.error("No valid sample directories found from CSV")
                raise ValueError("No valid sample directories found")
            
            # Split samples for training and validation
            train_size = int(len(sample_ids) * 0.8)  # 80% for training, 20% for validation
            train_ids = sample_ids[:train_size]
            val_ids = sample_ids[train_size:] if train_size < len(sample_ids) else sample_ids
            
            logger.info(f"Using {len(train_ids)} samples for training and {len(val_ids)} for validation")
            
            # Create the dataloaders
            logger.info(f"Creating training dataloader")
            self.train_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                sample_ids=train_ids,
                batch_size=train_loader_config.get('batch_size', 4),
                shuffle=train_loader_config.get('shuffle', True),
                num_workers=train_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                use_tiff=use_tiff,
                input_dir=input_dir,
                target_dir=target_dir
            )
            
            logger.info(f"Creating validation dataloader")
            self.val_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                sample_ids=val_ids,
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=val_loader_config.get('shuffle', False),
                num_workers=val_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                use_tiff=use_tiff,
                input_dir=input_dir,
                target_dir=target_dir
            )
            
            logger.info(f"Created dataloader with {len(self.train_loader.dataset)} training samples and {len(self.val_loader.dataset)} validation samples")
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
        trainer.run()
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