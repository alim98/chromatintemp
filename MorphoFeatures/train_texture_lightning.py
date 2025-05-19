#!/usr/bin/env python
"""
Wrapper script to train texture models using the Lightning implementation.
This script makes it easier to run the modernized texture training.
"""
import os
import sys
import argparse

# Add the parent directory to the path so we can import MorphoFeatures
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

# Import the Lightning trainer
from morphofeatures.texture.texture_lightning import TextureNet, make_loaders

def main():
    parser = argparse.ArgumentParser(description="Train MorphoFeatures texture model using Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="lightning_output", help="Output directory")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import required Lightning modules
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set output directory in config
    config['project_directory'] = args.output_dir
    
    # Create data loaders
    train_loader, val_loader = make_loaders(args.config)
    
    # Create model
    model = TextureNet(config)
    
    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Configure logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, 'logs'),
        name='texture_model'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        max_epochs=config.get('num_epochs', 50),
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed" if args.amp else "32",
        default_root_dir=args.output_dir
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Model saved at: {trainer.checkpoint_callback.best_model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 