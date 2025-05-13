import os
import sys
import argparse
import yaml
import torch.nn as nn

# Add MorphoFeatures to the path
sys.path.append(os.path.abspath("MorphoFeatures"))

# We'll try to import from MorphoFeatures, but we'll use compatibility wrappers
try:
    # Import necessary components from MorphoFeatures
    from MorphoFeatures.morphofeatures.texture.train import compile_criterion, set_up_training
    from MorphoFeatures.morphofeatures.texture.cell_loader import collate_contrastive
except ImportError as e:
    print(f"Warning: Could not import from MorphoFeatures texture module: {e}")
    print("Make sure MorphoFeatures is properly installed.")
    sys.exit(1)

# Import our custom dataloaders
from dataloader.highres_contrastive_dataloader import get_highres_contrastive_loaders
from dataloader.lowres_contrastive_dataloader import get_lowres_contrastive_loaders
from dataloader.contrastive_transforms import collate_contrastive


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
    
    # Load the trainer
    if from_checkpoint:
        try:
            from inferno.trainers.basic import Trainer
            trainer = Trainer().load(from_directory=project_directory,
                                    filename='Weights/checkpoint.pytorch')
        except ImportError:
            logger.error("Could not import Trainer from inferno. Make sure it's installed.")
            return
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return
    else:
        # Load config and set up training
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        try:
            # Import necessary libraries
            import torch.nn as nn
            import neurofire.models as models
            from inferno.trainers.basic import Trainer
            
            # Set up the trainer
            trainer = set_up_training(project_directory, config)
        except ImportError:
            logger.error("Could not import required libraries (inferno, neurofire).")
            return
        except Exception as e:
            logger.error(f"Error setting up training: {e}")
            return
    
    # Load our custom dataloaders
    logger.info(f"Loading training and validation data loader from {config_file}")
    loader = CustomTextureLoader(config_file, texture_type=texture_type)
    train_loader, validation_loader = loader.get_train_loaders()
    
    # Set max number of epochs
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    trainer.set_max_num_epochs(config.get('num_epochs', 10))
    
    # Bind loaders to trainer
    logger.info("Binding loaders to trainer")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)
    
    # GPU setup
    if isinstance(trainer.model, torch.nn.DataParallel):
        trainer.model = trainer.model.module
    trainer.cuda([0])  # Use first GPU
    
    # Set optimization level if using AMP
    trainer.apex_opt_level = config.get('opt_level', "O1")
    trainer.mixed_precision = config.get('mixed_precision', "False")
    
    # Multi-GPU setup if needed
    if len(devices.split(',')) > 1:
        trainer.model = nn.DataParallel(trainer.model)
    
    # Use dill for pickle since it handles more types
    trainer.pickle_module = 'dill'
    
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