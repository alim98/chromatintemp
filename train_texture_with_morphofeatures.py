#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import torch.nn as nn
import logging
import torch

# Add MorphoFeatures to the path if needed
if not any("MorphoFeatures" in p for p in sys.path):
    sys.path.append(os.path.abspath("MorphoFeatures"))

# Use try-except to handle different import scenarios
try:
    from MorphoFeatures.morphofeatures.texture.train import create_unet_model, ModelTrainer, compile_criterion
except ImportError:
    try:
        from morphofeatures.texture.train import create_unet_model, ModelTrainer, compile_criterion
    except ImportError:
        print("Error: Unable to import from morphofeatures.texture.train")
        print("Current path:", sys.path)
        sys.exit(1)

# Import our custom dataloaders
from dataloader.highres_contrastive_dataloader import get_highres_contrastive_loaders
from dataloader.lowres_contrastive_dataloader import get_lowres_contrastive_loaders
from dataloader.contrastive_transforms import collate_contrastive as custom_collate_contrastive


class CustomTextureLoader:
    """
    A custom loader class that adapts our dataloaders to work with MorphoFeatures texture training.
    This mimics the CellLoaders class from MorphoFeatures.
    """
    def __init__(self, configuration_file, texture_type='coarse'):
        """
        Initialize the loader with a configuration file.
        
        Args:
            configuration_file (str): Path to the configuration file
            texture_type (str): Either 'coarse' for lowres or 'fine' for highres
        """
        with open(configuration_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.texture_type = texture_type
        print(f"Initializing texture loader for {texture_type} texture analysis")
        
        # Determine which loader to use based on texture type
        if texture_type == 'coarse':
            self.loader_func = get_lowres_contrastive_loaders
            self.target_size = (64, 64)
        else:  # 'fine'
            self.loader_func = get_highres_contrastive_loaders
            self.target_size = (224, 224)
    
    def get_train_loaders(self):
        """
        Get train and validation dataloaders.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        data_config = self.config['data_config']
        
        loaders = self.loader_func(
            root_dir=data_config['root_dir'],
            batch_size=self.config['loader_config'].get('batch_size', 8),
            shuffle=self.config['loader_config'].get('shuffle', True),
            num_workers=self.config['loader_config'].get('num_workers', 4),
            class_csv_path=data_config.get('class_csv_path'),
            target_size=self.target_size,
            z_window_size=5,  # Default, adjust based on your data
            pin_memory=self.config['loader_config'].get('pin_memory', True),
            debug=True  # Set to False for production
        )
        
        train_loader = loaders['train'] 
        val_loader = loaders['val']
        
        return train_loader, val_loader
    
    def get_predict_loaders(self):
        """
        Get a dataloader for prediction.
        
        Returns:
            DataLoader: Prediction dataloader
        """
        data_config = self.config['data_config']
        
        # Create a single dataset for prediction (no contrastive pairs)
        if self.texture_type == 'coarse':
            from dataloader.lowres_image_dataloader import get_lowres_image_dataloader
            
            pred_loader = get_lowres_image_dataloader(
                root_dir=data_config['root_dir'],
                batch_size=self.config['pred_loader_config'].get('batch_size', 1),
                shuffle=False,
                num_workers=self.config['pred_loader_config'].get('num_workers', 0),
                class_csv_path=data_config.get('class_csv_path'),
                target_size=self.target_size,
                pin_memory=self.config['pred_loader_config'].get('pin_memory', False)
            )
        else:
            from dataloader.highres_image_dataloader import get_highres_image_dataloader
            
            pred_loader = get_highres_image_dataloader(
                root_dir=data_config['root_dir'],
                batch_size=self.config['pred_loader_config'].get('batch_size', 1),
                shuffle=False,
                num_workers=self.config['pred_loader_config'].get('num_workers', 0),
                class_csv_path=data_config.get('class_csv_path'),
                target_size=self.target_size,
                pin_memory=self.config['pred_loader_config'].get('pin_memory', False)
            )
        
        return pred_loader


def training(project_directory, texture_type, config_file, devices, from_checkpoint=False):
    """
    Run texture model training using our custom dataloaders.
    
    Args:
        project_directory (str): Directory to save results
        texture_type (str): 'coarse' or 'fine'
        config_file (str): Path to configuration file
        devices (str): Device IDs (e.g., "0" for first GPU)
        from_checkpoint (bool): Whether to resume from checkpoint
    """
    import time
    import os
    import logging
    import torch
    
    # Set up logging
    logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading config from {config_file}")
    
    # Set device
    logger.info(f"Using devices {devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    
    # Initialize trainer as None
    trainer = None
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the trainer
    if from_checkpoint:
        try:
            # Create a basic model as placeholder (will be replaced with loaded weights)
            model_config = config.get('model_kwargs', {})
            model = create_unet_model(config)
            
            # Initialize trainer with the model
            trainer = ModelTrainer(model)
            
            # Load checkpoint
            trainer = trainer.load(
                from_directory=os.path.join(project_directory, 'Weights'),
                filename='checkpoint.pytorch'
            )
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return
    else:
        try:
            # Create model
            model = create_unet_model(config)
            
            # Create criterion
            criterion = compile_criterion(config.get('loss'), **config.get('loss_kwargs', {}))
            
            # Create trainer
            trainer = ModelTrainer(model)
            logger.info("Built trainer with model")
            
            # Add steps one by one with debugging
            try:
                trainer.build_criterion(criterion)
                logger.info("Built criterion successfully")
                
                trainer.build_validation_criterion(criterion)
                logger.info("Built validation criterion successfully")
                
                # Extract optimizer config
                optimizer_config = config.get('training_optimizer_kwargs', {})
                optimizer_name = optimizer_config.get('optimizer', 'Adam')
                # Ensure optimizer_name is a string, not an integer
                if isinstance(optimizer_name, int):
                    logger.warning(f"Found integer optimizer_name: {optimizer_name}, using 'Adam' instead")
                    optimizer_name = 'Adam'
                optimizer_params = optimizer_config.get('optimizer_kwargs', {})
                
                # Build optimizer with correct parameters
                trainer.build_optimizer(optimizer_name, **optimizer_params)
                logger.info("Built optimizer successfully")
                
                # Debug validate_every - potential issue here
                validate_every_val = (100, 'iterations')
                logger.info(f"validate_every: {validate_every_val} (type: {type(validate_every_val)})")
                logger.info(f"validate_every[0]: {validate_every_val[0]} (type: {type(validate_every_val[0])})")
                
                trainer.set_validate_every(validate_every_val, for_num_iterations=20)
                logger.info("Set validate_every successfully")
                
                trainer.save_every((1000, 'iterations'), 
                                  to_directory=os.path.join(project_directory, 'Weights'))
                logger.info("Set save_every successfully")
                
                # Set up tensorboard logging
                trainer.build_logger(log_directory=os.path.join(project_directory, 'Logs'))
                logger.info("Built logger successfully")
                
            except Exception as e:
                logger.error(f"Error during trainer setup: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
            
            # Print config for debugging
            logger.info("Training optimizer kwargs:")
            logger.info(f"Config: {config}")
            logger.info(f"Loss: {config.get('loss')}")
            logger.info(f"Loss kwargs: {config.get('loss_kwargs')}")
            logger.info(f"Optimizer config: {optimizer_config}")
            logger.info(f"Optimizer name: {optimizer_name} (type: {type(optimizer_name)})")
            logger.info(f"Optimizer params: {optimizer_params}")
            
            # Debug what's in the criterion
            try:
                criterion_debug = compile_criterion(config.get('loss'), **config.get('loss_kwargs', {}))
                logger.info(f"Criterion type: {type(criterion_debug)}")
            except Exception as e:
                logger.error(f"Error creating criterion: {e}")
            
        except Exception as e:
            logger.error(f"Error setting up training: {e}")
            return
    
    # Verify that trainer was properly initialized
    if trainer is None:
        logger.error("Trainer was not properly initialized. Aborting.")
        return
    
    # Load our custom dataloaders
    logger.info(f"Loading training and validation data loader from {config_file}")
    loader = CustomTextureLoader(config_file, texture_type=texture_type)
    train_loader, validation_loader = loader.get_train_loaders()
    
    # Set max number of epochs
    trainer.set_max_num_epochs(config.get('num_epochs', 10))
    
    # Bind loaders to trainer
    logger.info("Binding loaders to trainer")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)
    
    # GPU setup
    if torch.cuda.is_available():
        device_ids = [i for i in range(len(devices.split(',')))]
        if len(device_ids) > 0:
            trainer.cuda(device_ids=device_ids if len(device_ids) > 1 else None)
    
    # Start training
    logger.info("Starting training!")
    start = time.time()
    trainer.fit()
    end = time.time()
    time_diff = end - start
    logger.info(f"Training took {time_diff // 3600} hours {time_diff % 3600 // 60} minutes")


def main():
    parser = argparse.ArgumentParser(description="Train texture model with custom dataloaders")
    parser.add_argument('--project_dir', type=str, required=True, help="Project directory for results")
    parser.add_argument('--texture_type', type=str, choices=['coarse', 'fine'], default='coarse',
                        help="Type of texture analysis (coarse or fine)")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--devices', type=str, default='0', help="GPU device indices")
    parser.add_argument('--from_checkpoint', type=int, default=0, help="Resume from checkpoint (0/1)")
    args = parser.parse_args()
    
    # Create the project directory if it doesn't exist
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Start training
    training(args.project_dir, args.texture_type, args.config, args.devices, bool(args.from_checkpoint))


if __name__ == "__main__":
    main() 