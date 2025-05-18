import os
import sys
import torch
import yaml
import argparse
from tqdm import tqdm

# Add MorphoFeatures to the path
sys.path.append(os.path.abspath("MorphoFeatures"))

# Import the MorphoFeatures shape trainer
from MorphoFeatures.morphofeatures.shape.train_shape_model import ShapeTrainer

# Import our custom dataloader adapter
from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader


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
        print("Building custom dataloaders...")
        dataset_config = self.config['data']
        loader_config = self.config['loader']
        
        # Create train dataloader using our custom function
        self.train_loader = get_morphofeatures_mesh_dataloader(
            root_dir=dataset_config['root_dir'],
            class_csv_path=dataset_config.get('class_csv_path'),
            batch_size=loader_config.get('batch_size', 8),
            shuffle=loader_config.get('shuffle', True),
            num_workers=loader_config.get('num_workers', 4),
            precomputed_dir=dataset_config.get('precomputed_dir'),
            num_points=dataset_config.get('num_points', 1024),
            cache_dir=dataset_config.get('cache_dir'),
            debug=True  # Set to False for production
        )
        
        # Create validation dataloader
        if dataset_config.get('val_root_dir'):
            # If a separate validation directory is specified
            self.val_loader = get_morphofeatures_mesh_dataloader(
                root_dir=dataset_config['val_root_dir'],
                class_csv_path=dataset_config.get('val_class_csv_path', dataset_config.get('class_csv_path')),
                batch_size=loader_config.get('batch_size', 8),
                shuffle=False,
                num_workers=loader_config.get('num_workers', 4),
                precomputed_dir=dataset_config.get('precomputed_dir'),
                num_points=dataset_config.get('num_points', 1024),
                cache_dir=dataset_config.get('cache_dir'),
                debug=True  # Set to False for production
            )
        else:
            # Use the training data for validation (not ideal but works for small datasets)
            print("Warning: Using training data for validation. For proper evaluation, provide val_root_dir.")
            self.val_loader = self.train_loader


def main():
    parser = argparse.ArgumentParser(description="Train MorphoFeatures shape model with custom dataloaders")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer with our custom class
    trainer = CustomShapeTrainer(config)
    
    # Run training
    print("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main() 