#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import torch.nn as nn
import logging
import torch

if not any("MorphoFeatures" in p for p in sys.path):
    sys.path.append(os.path.abspath("MorphoFeatures"))

try:
    from MorphoFeatures.morphofeatures.texture.train import create_unet_model, ModelTrainer, compile_criterion
except ImportError:
    try:
        from morphofeatures.texture.train import create_unet_model, ModelTrainer, compile_criterion
    except ImportError:

        sys.exit(1)

from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader as get_lowres_texture_dataloader
from dataloader.highres_texture_adapter import get_morphofeatures_highres_texture_dataloader as get_highres_texture_dataloader


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
        
        # Get data configuration
        self.data_config = self.config.get('data_config', {})
        
        if texture_type == 'coarse':
            self.loader_func = get_lowres_texture_dataloader
            self.box_size = self.data_config.get('box_size', [104, 104, 104])
            self.target_size = (self.box_size[1], self.box_size[2])  # Use H,W from box_size
        else:  # 'fine'
            self.loader_func = get_highres_texture_dataloader
            self.box_size = self.data_config.get('box_size', [256, 256, 256])
            self.target_size = (self.box_size[1], self.box_size[2])  # Use H,W from box_size
        
        # Debug output
        print(f"Using box_size: {self.box_size}, target_size: {self.target_size}")
        print(f"Data root directory: {self.data_config.get('root_dir')}")
        print(f"Using CSV file: {self.data_config.get('class_csv_path')}")
    
    def get_train_loaders(self):
        """
        Get train and validation dataloaders.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Extract all relevant config values
        root_dir = self.data_config.get('root_dir')
        class_csv_path = self.data_config.get('class_csv_path')
        is_cytoplasm = self.data_config.get('is_cytoplasm', False)
        use_tiff = self.data_config.get('use_tiff', True)
        input_dir = self.data_config.get('input_dir', 'raw')
        target_dir = self.data_config.get('target_dir', 'mask')
        
        # Get loader configs
        loader_config = self.config.get('loader_config', {})
        val_loader_config = self.config.get('val_loader_config', {})
        
        # Print debug info to help diagnose sample loading issues
        print(f"Loading samples from {root_dir}")
        print(f"Using CSV file: {class_csv_path}")
        
        # # List directories to verify they exist
        # if os.path.exists(root_dir):
        #     print(f"Root directory exists with contents: {os.listdir(root_dir)[:10]}...")
        #     # If we expect a class structure
        #     if self.texture_type == 'coarse':
        #         for class_dir in os.listdir(root_dir):
        #             class_path = os.path.join(root_dir, class_dir)
        #             if os.path.isdir(class_path):
        #                 sample_count = len([d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))])
        #                 print(f"  - Class {class_dir}: {sample_count} samples")
        # else:
        #     print(f"WARNING: Root directory {root_dir} does not exist!")
        
        # Check if CSV file exists
        if class_csv_path and os.path.exists(class_csv_path):
            import pandas as pd
            df = pd.read_csv(class_csv_path)
            print(f"CSV file contains {len(df)} rows and {len(df['sample_id'].unique())} unique samples")
            print(f"CSV columns: {df.columns.tolist()}")
        else:
            print(f"WARNING: CSV file {class_csv_path} does not exist or is not specified!")
        
        # Determine full sample set
        if self.texture_type == 'coarse':
            # For lowres: discover from directory structure
            all_samples = []
            for class_name in os.listdir(root_dir):
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    for sample_id in os.listdir(class_path):
                        sample_path = os.path.join(class_path, sample_id)
                        raw_dir = os.path.join(sample_path, input_dir)
                        mask_dir = os.path.join(sample_path, target_dir)
                        if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                            all_samples.append(sample_path)
        else:
            # For highres: discover from CSV
            all_samples = []
            if class_csv_path and os.path.exists(class_csv_path):
                import pandas as pd
                df = pd.read_csv(class_csv_path)
                for sample_id in df['sample_id'].unique():
                    sample_path = os.path.join(root_dir, str(sample_id).zfill(4))
                    raw_dir = os.path.join(sample_path, input_dir)
                    mask_dir = os.path.join(sample_path, target_dir)
                    if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                        all_samples.append(sample_path)
        
        print(f"Found {len(all_samples)} total valid samples")
        
        # Split samples for training (80%) and validation (20%)
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        print(f"Using {len(train_samples)} samples for training and {len(val_samples)} for validation")
        
        # Create training dataloader - use different parameters based on texture type
        if self.texture_type == 'coarse':
            # Use box_size for lowres
            train_loader = self.loader_func(
                root_dir=root_dir,
                sample_ids=train_samples,
                batch_size=loader_config.get('batch_size', 4),
                shuffle=loader_config.get('shuffle', True),
                num_workers=loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=self.box_size,
                use_tiff=use_tiff,
                input_dir=input_dir,
                target_dir=target_dir
            )
            
            # Create validation dataloader
            val_loader = self.loader_func(
                root_dir=root_dir,
                sample_ids=val_samples,
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=val_loader_config.get('shuffle', False),
                num_workers=val_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                box_size=self.box_size,
                use_tiff=use_tiff,
                input_dir=input_dir,
                target_dir=target_dir
            )
        else:
            # For highres, extract just the sample_ids from the paths
            train_sample_ids = []
            for path in train_samples:
                # Extract the sample ID from the path: /path/to/0001 -> "0001"
                sample_id = os.path.basename(path)
                train_sample_ids.append(sample_id)
            
            val_sample_ids = []
            for path in val_samples:
                sample_id = os.path.basename(path)
                val_sample_ids.append(sample_id)
                
            # Debug to verify sample IDs
            print(f"First 5 training sample IDs: {train_sample_ids[:5]}")
            
            # Use cube_size for highres
            cube_size = self.box_size[0]  # Use the first dimension as cube size (they should all be equal)
            train_loader = self.loader_func(
                root_dir=root_dir,
                sample_ids=train_sample_ids,  # Use just the IDs, not full paths
                batch_size=loader_config.get('batch_size', 4),
                shuffle=loader_config.get('shuffle', True),
                num_workers=loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                cube_size=cube_size,  # Use cube_size instead of box_size
                class_csv_path=class_csv_path,
                debug=True
            )
            
            # Create validation dataloader
            val_loader = self.loader_func(
                root_dir=root_dir,
                sample_ids=val_sample_ids,  # Use just the IDs, not full paths
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=val_loader_config.get('shuffle', False),
                num_workers=val_loader_config.get('num_workers', 0),
                is_cytoplasm=is_cytoplasm,
                cube_size=cube_size,  # Use cube_size instead of box_size
                class_csv_path=class_csv_path,
                debug=True
            )
        
        print(f"Created dataloader with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")
        
        return train_loader, val_loader
    
    def get_predict_loaders(self):
        """
        Get a dataloader for prediction.
        
        Returns:
            DataLoader: Prediction dataloader
        """
        # Since we're only concerned with training for now, just return a small subset
        # of the training data for prediction/inference
        train_loader, _ = self.get_train_loaders()
        
        # Get pred loader config
        pred_loader_config = self.config.get('pred_loader_config', {})
        
        # Create a new loader with prediction settings
        root_dir = self.data_config.get('root_dir')
        class_csv_path = self.data_config.get('class_csv_path')
        
        if self.texture_type == 'coarse':
            from dataloader.lowres_image_dataloader import get_lowres_image_dataloader
            
            pred_loader = get_lowres_image_dataloader(
                root_dir=root_dir,
                batch_size=pred_loader_config.get('batch_size', 1),
                shuffle=False,
                num_workers=pred_loader_config.get('num_workers', 0),
                class_csv_path=class_csv_path,
                target_size=self.target_size,
                z_window_size=self.box_size[0],  # Use first dimension from box_size
                pin_memory=pred_loader_config.get('pin_memory', False)
            )
        else:
            from dataloader.highres_image_dataloader import get_highres_image_dataloader
            
            pred_loader = get_highres_image_dataloader(
                root_dir=root_dir,
                batch_size=pred_loader_config.get('batch_size', 1),
                shuffle=False,
                num_workers=pred_loader_config.get('num_workers', 0),
                class_csv_path=class_csv_path,
                target_size=self.target_size,
                z_window_size=self.box_size[0],  # Use first dimension from box_size
                pin_memory=pred_loader_config.get('pin_memory', False)
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
    
    # logger.info(f"Loading config from {config_file}")
    
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
    trainer.set_max_num_epochs(config.get('num_epochs', 50))
    
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