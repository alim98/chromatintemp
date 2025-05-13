#!/usr/bin/env python
import os
import sys
import torch
import yaml
import argparse
import logging
import time
from tqdm import tqdm

# Add MorphoFeatures to the path
sys.path.append(os.path.abspath("MorphoFeatures"))

# Import MorphoFeatures shape components
from MorphoFeatures.morphofeatures.shape.train_shape_model import ShapeTrainer

# Import custom dataloaders
from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader

# Setup logging
logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s',
                   stream=sys.stdout,
                   level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomShapeTrainer(ShapeTrainer):
    """
    A custom trainer that extends MorphoFeatures' ShapeTrainer to use our dataloaders.
    """
    def __init__(self, config):
        # Initialize the parent class
        super().__init__(config)
    
    def build_loaders(self):
        """
        Override the build_loaders method to use our custom dataloaders.
        """
        print("Building custom shape dataloaders...")
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
            debug=False  # Set to False for production
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
                debug=False  # Set to False for production
            )
        else:
            # Use the training data for validation (not ideal but works for small datasets)
            print("Warning: Using training data for validation. For proper evaluation, provide val_root_dir.")
            self.val_loader = self.train_loader


class TextureModelTrainer:
    """
    Trainer for texture models (both low-res and high-res)
    """
    def __init__(self, config, model_type='lowres'):
        # Try to import neurofire and inferno libraries needed for texture models
        try:
            # First try to import the core dependencies
            import neurofire.models as texture_models
            from inferno.trainers.basic import Trainer
            from inferno.trainers.callbacks.scheduling import AutoLR
            from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore, GarbageCollection
            
            # Save core dependencies
            self.texture_models = texture_models
            self.Trainer = Trainer
            self.AutoLR = AutoLR
            self.SaveAtBestValidationScore = SaveAtBestValidationScore
            self.GarbageCollection = GarbageCollection
            
            # Try to import optional dependencies
            try:
                from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
                self.TensorboardLogger = TensorboardLogger
                self.has_tensorboard = True
            except ImportError:
                print("TensorboardLogger not available - continuing without logging")
                self.has_tensorboard = False
            
            # Try to import Upsample (for patching)
            try:
                from inferno.extensions.layers.sampling import Upsample
                import torch.nn as nn
                
                # Create a custom Upsample3D class that uses trilinear instead of bilinear
                class Upsample3D(nn.Module):
                    def __init__(self, scale_factor=2, mode='trilinear', align_corners=False):
                        super(Upsample3D, self).__init__()
                        self.scale_factor = scale_factor
                        self.mode = mode
                        self.align_corners = align_corners
                    
                    def forward(self, input):
                        return nn.functional.interpolate(input, scale_factor=self.scale_factor, 
                                                      mode=self.mode, align_corners=self.align_corners)
                
                # Monkey patch the Upsample class to use Upsample3D for 5D inputs
                original_upsample_forward = Upsample.forward
                def patched_forward(self, input):
                    if input.dim() == 5:  # If 5D input (batch, channels, depth, height, width)
                        return nn.functional.interpolate(input, scale_factor=self.scale_factor, 
                                                      mode='trilinear')
                    else:
                        return original_upsample_forward(self, input)
                
                Upsample.forward = patched_forward
            except ImportError:
                print("Could not patch Upsample class - model may fail with 5D inputs")
            
            # Try to import the custom dataloader adapter
            try:
                from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader
                self.get_morphofeatures_texture_dataloader = get_morphofeatures_texture_dataloader
                self.has_texture_dataloader = True
            except ImportError as e:
                print(f"Could not import texture dataloader adapter: {e}")
                self.has_texture_dataloader = False
                
        except ImportError as e:
            logger.error("Required libraries for texture model training are not installed.")
            logger.error("Please install neurofire and inferno: pip install neurofire inferno-pytorch")
            raise
        
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.get('device', 'cpu'))
        self.project_dir = config.get('project_directory', f'experiments/{model_type}_texture_model')
        os.makedirs(os.path.join(self.project_dir, 'Weights'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'Logs'), exist_ok=True)
        
        # Setup the texture model
        self.setup_model()
        
        # Setup dataloaders
        self.setup_dataloaders()
    
    def setup_model(self):
        """Set up the texture model, criterion, optimizer, and trainer"""
        model_name = self.config.get('model_name', 'UNet3D')
        model_kwargs = self.config.get('model_kwargs', {})
        
        # Create the model
        model = getattr(self.texture_models, model_name)(**model_kwargs)
        
        # Compile the criterion
        criterion_name = self.config.get('loss', 'MSELoss')
        criterion_kwargs = self.config.get('loss_kwargs', {})
        
        if isinstance(criterion_name, str):
            criterion = getattr(torch.nn, criterion_name)(**criterion_kwargs)
        else:
            # Handle more complex criterion setup if needed
            criterion = None
        
        # Setup trainer
        logger.info("Building texture trainer")
        smoothness = self.config.get('smoothness', 0.95)
        
        # Create trainer with model
        trainer = self.Trainer(model)
        
        # Build criterion
        trainer.build_criterion(criterion)
        
        # Build optimizer
        optimizer_config = self.config.get('training_optimizer_kwargs', {})
        optimizer_method = optimizer_config.pop('optimizer', 'Adam')
        optimizer_kwargs = optimizer_config.pop('optimizer_kwargs', {}) if 'optimizer_kwargs' in optimizer_config else {}
        trainer.build_optimizer(optimizer_method, **optimizer_kwargs)
        
        # Set up save directory and interval
        trainer.save_every((1000, 'iterations'), to_directory=os.path.join(self.project_dir, 'Weights'))
        
        # Set up validation
        trainer.validate_every((100, 'iterations'), for_num_iterations=20)
        
        # Register callbacks
        trainer.register_callback(self.SaveAtBestValidationScore(smoothness=smoothness, verbose=True))
        trainer.register_callback(self.AutoLR(factor=0.98,
                                 patience='100 iterations',
                                 monitor='validation_loss_averaged',
                                 monitor_while='validating',
                                 monitor_momentum=smoothness,
                                 verbose=True))
        trainer.register_callback(self.GarbageCollection())
        
        # Skip TensorboardLogger since it requires TensorFlow
        # Instead, print a message explaining that no logging will be used
        print("Skipping TensorBoard logging since TensorFlow is not installed.")
        
        # Move to device
        trainer.cuda([0]) if self.config.get('device') == 'cuda' else trainer.cpu()
        
        # Set mixed precision settings if available
        trainer.apex_opt_level = self.config.get('opt_level', "O1")
        trainer.mixed_precision = self.config.get('mixed_precision', "False")
        
        self.trainer = trainer
    
    def setup_dataloaders(self):
        """Setup custom texture dataloaders for training and validation"""
        print(f"Setting up {self.model_type} texture dataloaders")
        
        # Get dataset config
        data_config = self.config.get('data_config', {})
        root_dir = data_config.get('root_dir', 'data')
        class_csv_path = data_config.get('class_csv_path', None)
        
        # Get specific config for texture data
        is_cytoplasm = data_config.get('is_cytoplasm', False)
        box_size = data_config.get('box_size', (104, 104, 104))
        split = data_config.get('split', 0.2)
        seed = data_config.get('seed', 42)
        
        # Get loader configs
        train_loader_config = self.config.get('loader_config', {})
        val_loader_config = self.config.get('val_loader_config', {})
        
        # Check if custom dataloader is available
        if not hasattr(self, 'has_texture_dataloader') or not self.has_texture_dataloader:
            raise ImportError("Custom texture dataloader is not available. Install required dependencies.")
        
        # Load all sample IDs from CSV
        import pandas as pd
        if class_csv_path and os.path.exists(class_csv_path):
            df = pd.read_csv(class_csv_path)
            sample_ids = df['sample_id'].astype(str).tolist()
            
            # Split the data
            import numpy as np
            np.random.seed(seed)
            np.random.shuffle(sample_ids)
            split_idx = int(len(sample_ids) * (1 - split))
            train_sample_ids = sample_ids[:split_idx]
            val_sample_ids = sample_ids[split_idx:]
            
            print(f"Total samples: {len(sample_ids)}")
            print(f"Training samples: {len(train_sample_ids)}")
            print(f"Validation samples: {len(val_sample_ids)}")
            
            # Create training dataloader
            self.train_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                batch_size=train_loader_config.get('batch_size', 4),
                shuffle=train_loader_config.get('shuffle', True),
                num_workers=train_loader_config.get('num_workers', 4),
                class_csv_path=class_csv_path,
                sample_ids=train_sample_ids,
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                pin_memory=train_loader_config.get('pin_memory', True),
                debug=False
            )
            
            # Create validation dataloader
            self.val_loader = self.get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=val_loader_config.get('shuffle', False),
                num_workers=val_loader_config.get('num_workers', 4),
                class_csv_path=class_csv_path,
                sample_ids=val_sample_ids,
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                pin_memory=val_loader_config.get('pin_memory', True),
                debug=False
            )
        else:
            raise ValueError("Class CSV file not found. A valid class CSV file is required to create dataloaders.")
    
    def train(self):
        """Train the texture model"""
        logger.info(f"Starting {self.model_type} texture model training")
        
        # Bind dataloaders to trainer
        self.trainer.bind_loader('train', self.train_loader)
        self.trainer.bind_loader('validate', self.val_loader)
        
        # Set max epochs
        self.trainer.set_max_num_epochs(self.config.get('num_epochs', 10))
        
        # Train the model
        start = time.time()
        self.trainer.fit()
        end = time.time()
        
        # Report training time
        time_diff = end - start
        print(f"The {self.model_type} texture training took {time_diff // 3600} hours {time_diff % 3600 // 60} minutes")


def train_shape_model(config_path):
    """Train a shape model using the custom shape trainer"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Training shape model...")
    trainer = CustomShapeTrainer(config)
    trainer.run()
    print("Shape model training complete!")


def train_texture_model(config_path, model_type='lowres'):
    """Train a texture model (lowres or highres) using the texture trainer"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Training {model_type} texture model...")
        
        # Use the PyTorch-based implementation instead of neurofire/inferno
        try:
            from pytorch_texture_model import PyTorchTextureTrainer
            
            print("Using PyTorch-based texture model trainer")
            trainer = PyTorchTextureTrainer(config, model_type=model_type)
            trainer.train()
            
        except ImportError as e:
            # Fall back to the original implementation if the new one is not available
            logger.warning(f"Could not use PyTorch texture trainer: {e}")
            logger.warning("Falling back to original neurofire/inferno implementation...")
            
            trainer = TextureModelTrainer(config, model_type=model_type)
            trainer.train()
        
        print(f"{model_type.capitalize()} texture model training complete!")
        
    except ImportError as e:
        logger.error(f"Could not train texture model: {e}")
        logger.info("Skipping texture model training due to missing dependencies.")


def main():
    parser = argparse.ArgumentParser(description="Train MorphoFeatures models (shape, lowres, highres)")
    parser.add_argument("--model", type=str, required=True, choices=['shape', 'lowres', 'highres', 'all'],
                      help="Model type to train (shape, lowres, highres, or all)")
    parser.add_argument("--config", type=str, required=True, 
                      help="Path to config file (or config directory for 'all')")
    args = parser.parse_args()
    
    if args.model == 'shape':
        train_shape_model(args.config)
    
    elif args.model == 'lowres':
        train_texture_model(args.config, model_type='lowres')
    
    elif args.model == 'highres':
        train_texture_model(args.config, model_type='highres')
    
    elif args.model == 'all':
        # Assuming config is a directory containing all configuration files
        config_dir = args.config
        shape_config = os.path.join(config_dir, 'shape_config.yaml')
        lowres_config = os.path.join(config_dir, 'lowres_texture_config.yaml')
        highres_config = os.path.join(config_dir, 'highres_texture_config.yaml')
        
        # Train all models in sequence
        print("Training all MorphoFeatures models...")
        
        print("\n=== Starting Shape Model Training ===")
        train_shape_model(shape_config)
        
        try:
            print("\n=== Starting Low-Resolution Texture Model Training ===")
            train_texture_model(lowres_config, model_type='lowres')
            
            print("\n=== Starting High-Resolution Texture Model Training ===")
            train_texture_model(highres_config, model_type='highres')
        except ImportError:
            logger.warning("Texture model training skipped due to missing dependencies.")
        
        print("\n=== MorphoFeatures model training completed! ===")


if __name__ == "__main__":
    main() 