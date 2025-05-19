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
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Add MorphoFeatures to the path
sys.path.append(os.path.abspath("MorphoFeatures"))
logger.info("Added MorphoFeatures to the path")

# Import shape modules
from MorphoFeatures.morphofeatures.shape.train_shape_model import ShapeTrainer
from MorphoFeatures.morphofeatures.shape.network import DeepGCN

# Import neural network modules
from MorphoFeatures.morphofeatures.nn.texture_encoder import TextureEncoder
from MorphoFeatures.morphofeatures.nn.losses import get_shape_loss, get_texture_loss

# Import the new Lightning implementation
try:
    # Try direct import first
    from MorphoFeatures.morphofeatures.texture.texture_lightning import TextureNet
except ImportError:
    # If that fails, try to import as a module
    try:
        from MorphoFeatures.morphofeatures import TextureNet
    except ImportError:
        logger.error("Failed to import TextureNet. Check your import paths.")
        sys.exit(1)

logger.info("Successfully imported MorphoFeatures modules")


# Import custom dataloaders
from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader
logger.info("Successfully imported mesh dataloader")
    

class CustomShapeTrainer(ShapeTrainer):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.ckpt_dir = os.path.join(config['experiment_dir'], 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.build_loaders()
        self.reset()
        # Custom initialization for wandb
        self.step = 0
        self.use_wandb = config.get("use_wandb", False)
        # Don't import wandb here - it will be imported only when needed

    # Override build_loaders to use our custom dataloader adapter
    def build_loaders(self):
        """Build data loaders using our custom adapter"""
        dataset_config = self.config['data']
        loader_config = self.config['loader']
        
        # Use our custom adapter instead of the default MorphoFeatures loader
        train_loader = get_morphofeatures_mesh_dataloader(
            root_dir=dataset_config['root_dir'],
            batch_size=loader_config['batch_size'],
            shuffle=True,
            num_workers=loader_config.get('num_workers', 4),
            class_csv_path=dataset_config.get('class_csv_path'),
            sample_ids=dataset_config.get('train_sample_ids'),
            filter_by_class=dataset_config.get('filter_by_class'),
            ignore_unclassified=dataset_config.get('ignore_unclassified', True),
            precomputed_dir=dataset_config.get('precomputed_dir'),
            generate_on_load=dataset_config.get('generate_on_load', True),
            num_points=dataset_config.get('num_points', 1024),
            sample_percent=dataset_config.get('sample_percent', 100),
            cache_dir=dataset_config.get('cache_dir'),
            pin_memory=loader_config.get('pin_memory', False),
            debug=dataset_config.get('debug', False)
        )
        
        # For validation loader, use the same parameters but different sample IDs
        val_loader = get_morphofeatures_mesh_dataloader(
            root_dir=dataset_config['root_dir'],
            batch_size=loader_config['batch_size'],
            shuffle=False,  # No shuffling for validation
            num_workers=loader_config.get('num_workers', 4),
            class_csv_path=dataset_config.get('class_csv_path'),
            sample_ids=dataset_config.get('val_sample_ids'),
            filter_by_class=dataset_config.get('filter_by_class'),
            ignore_unclassified=dataset_config.get('ignore_unclassified', True),
            precomputed_dir=dataset_config.get('precomputed_dir'),
            generate_on_load=dataset_config.get('generate_on_load', True),
            num_points=dataset_config.get('num_points', 1024),
            sample_percent=dataset_config.get('sample_percent', 100),
            cache_dir=dataset_config.get('cache_dir'),
            pin_memory=loader_config.get('pin_memory', False),
            debug=dataset_config.get('debug', False)
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training epoch {self.epoch}")):
            # unpack
            if isinstance(batch, dict):
                points = batch["points"].to(self.device)
                features = batch.get("features")
                if features is not None:
                    features = features.to(self.device)
            else:
                points, features = batch[0].to(self.device), None

            # forward
            out, h = self.model(points, features) if features is not None else self.model(points)

            # build labels
            N = out.size(0)
            if N % 2 == 0:
                labels = torch.arange(N // 2).repeat_interleave(2).to(self.device)
            else:
                labels = torch.arange(N).to(self.device)

            # loss + backward
            loss = self.criterion(out, labels)
            if torch.isnan(loss).item():
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Logging to wandb if enabled
            if self.use_wandb:
                try:
                    import wandb
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/epoch": self.epoch
                        },
                        step=self.step
                    )
                except Exception as e:
                    logger.warning(f"Wandb logging error: {e}")

            self.step += 1

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {self.epoch} → avg train loss: {avg_loss:.6f}")

        if self.use_wandb:
            try:
                import wandb
                wandb.log(
                    {
                        "train/avg_loss": avg_loss,
                        "train/epoch": self.epoch
                    },
                    step=self.step
                )
            except Exception as e:
                logger.warning(f"Wandb logging error: {e}")

        return avg_loss


    def validate_epoch(self):
        # only run validation on the prescribed schedule
        if self.epoch % self.config["training"]["validate_every"] != 0:
            return

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                if isinstance(batch, dict):
                    points = batch["points"].to(self.device)
                    features = batch.get("features")
                    if features is not None:
                        features = features.to(self.device)
                else:
                    points, features = batch[0].to(self.device), None

                out, h = self.model(points, features) if features is not None else self.model(points)

                N = out.size(0)
                if N % 2 == 0:
                    labels = torch.arange(N // 2).repeat_interleave(2).to(self.device)
                else:
                    labels = torch.arange(N).to(self.device)

                loss = self.criterion(out, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {self.epoch} → avg validation loss: {avg_loss:.6f}")

        if self.use_wandb:
            try:
                import wandb
                wandb.log(
                    {
                        "val/loss": avg_loss,
                        "val/epoch": self.epoch
                    },
                    step=self.step
                )
            except Exception as e:
                logger.warning(f"Wandb logging error: {e}")

        # checkpoint best
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
        use_wandb = self.config.get('use_wandb', True)
        self.use_wandb = use_wandb
        
        # We don't need to initialize wandb here since it's now handled in train_shape_model_from_config
        self.validate_epoch()
        self.train()
            
    def fit(self):
        """Wrapper method to match the expected API in train_morphofeatures_models.py"""
        logger.info("Starting model training via fit()...")
        self.run()


class TextureModelTrainer:
    """
    Trainer for texture models (both low-res and high-res) using PyTorch Lightning
    """
    def __init__(self, config, model_type='lowres'):
        logger.info(f"Initializing {model_type} TextureModelTrainer")
        # Using PyTorch Lightning and MONAI instead of inferno/neurofire
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.get('device', 'cpu'))
        logger.info(f"Using device: {self.device}")
        print(self.device)
        self.project_dir = config.get('project_directory', f'experiments/{model_type}_texture_model')
        os.makedirs(os.path.join(self.project_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'logs'), exist_ok=True)
        logger.info(f"Created project directory: {self.project_dir}")
        
        # Start a new wandb run for this training
        self.wandb_run = None
        if config.get('use_wandb', False):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    entity=config.get('wandb_entity', None),
                    project=config.get('wandb_project', 'MorphoFeatures'),
                    config={
                        'model_type': model_type,
                        'learning_rate': config.get('optimizer_config', {}).get('lr', 1e-4),
                        'architecture': 'TextureNet_Lightning',
                        'dataset': config.get('data_config', {}).get('root_dir', ''),
                        'epochs': config.get('num_epochs', 1),
                        'batch_size': config.get('loader_config', {}).get('batch_size', 4),
                        'box_size': config.get('data_config', {}).get('box_size', None),
                    }
                )
                logger.info("Started wandb run for training")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                logger.warning("Continuing without wandb logging")
        
        # Try to import the custom dataloader adapter
        from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader
        self.get_morphofeatures_texture_dataloader = get_morphofeatures_texture_dataloader
        self.has_texture_dataloader = True
        logger.info("Successfully imported texture dataloader")
            
        # Setup the texture model and dataloaders
        self.setup_model()
        self.setup_dataloaders()

    def setup_model(self):
        """Set up the Lightning texture model"""
        logger.info("Setting up Lightning model")
        
        # Adapt config format for TextureNet
        model_config = self.config.get('model_config', {})
        if 'f_maps' in self.config.get('model_kwargs', {}):
            model_config['feature_dim'] = self.config.get('model_kwargs', {}).get('out_channels', 80)
        
        # Extract loss configuration parameters
        loss_config = self.config.get('loss_config', {})
        lambda_nt_xent = self.config.get('lambda_nt_xent', 1.0)
        lambda_mse = self.config.get('lambda_mse', 0.5)
        lambda_reg = self.config.get('lambda_reg', 0.1)
        temperature = self.config.get('temperature', 0.1)
        
        # Create a copy of the config with updated format for Lightning
        lightning_config = {
            'model_config': model_config,
            'optimizer_config': {
                'lr': self.config.get('training_optimizer_kwargs', {}).get('optimizer_kwargs', {}).get('lr', 1e-4),
                'weight_decay': self.config.get('training_optimizer_kwargs', {}).get('optimizer_kwargs', {}).get('weight_decay', 1e-4)
            },
            'scheduler_config': {
                'type': 'plateau',
                'factor': 0.95,
                'patience': 5
            },
            # Loss configuration for NT-Xent + MSE + regularization
            'loss_config': {
                'type': 'morphofeatures',
                'temperature': temperature,
                'lambda_ae': lambda_mse,
                'lambda_norm': lambda_reg
            },
            'temperature': temperature,
            'lambda_rec': lambda_mse
        }
        
        # Create Lightning model
        self.model = TextureNet(lightning_config)
        logger.info(f"Created TextureNet Lightning model with MorphoFeaturesLoss (NT-Xent={lambda_nt_xent}, MSE={lambda_mse}, Reg={lambda_reg})")
        
        # Create Trainer
        self.trainer = None  # Will be created after dataloaders are set up
        
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
            
            # Decide dataloader logic based on model_type
            if self.model_type == 'lowres':
                # Directory-based sample discovery for lowres
                sample_ids = []
                for class_name in os.listdir(root_dir):
                    class_path = os.path.join(root_dir, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    for sample_id in os.listdir(class_path):
                        sample_path = os.path.join(class_path, sample_id)
                        if os.path.isdir(sample_path):
                            raw_dir = os.path.join(sample_path, input_dir)
                            mask_dir = os.path.join(sample_path, target_dir)
                            if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                                sample_ids.append(sample_path)
            else:
                # CSV-based sample discovery for highres
                sample_ids = []
                if class_csv_path and os.path.exists(class_csv_path):
                    df = pd.read_csv(class_csv_path, dtype={'sample_id': str})
                    if 'sample_id' in df.columns:
                        csv_samples = [str(id) for id in df['sample_id'].unique()]
                        for sample_id in csv_samples:
                            sample_path = os.path.join(root_dir, sample_id)
                            raw_dir = os.path.join(sample_path, input_dir)
                            mask_dir = os.path.join(sample_path, target_dir)
                            if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                                sample_ids.append(sample_path)
                    else:
                        logger.error("CSV does not contain 'sample_id' column")
                else:
                    logger.error("CSV file not found: {}".format(class_csv_path))

            if not sample_ids:
                logger.error("No valid sample directories found for {}".format(self.model_type))
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
            
            # Now that we have dataloaders, create the Lightning trainer
            self._setup_trainer()
            
        except Exception as e:
            logger.error(f"Error in setup_dataloaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _setup_trainer(self):
        """Set up the Lightning Trainer"""
        # Configure callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(self.project_dir, 'checkpoints'),
                filename='{epoch}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_last=True
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Add detailed loss logging callback
        class DetailedLossLogger(pl.Callback):
            def __init__(self, use_wandb=False):
                super().__init__()
                self.use_wandb = use_wandb
                
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # Log component losses if they're available in the output
                if isinstance(outputs, dict) and 'loss_components' in outputs:
                    components = outputs['loss_components']
                    for name, value in components.items():
                        # Log to progress bar for important components
                        if name in ['nt_xent', 'recon', 'reg']:
                            trainer.logger.log_metrics({f"train/loss_{name}": value}, step=trainer.global_step)
                        
                        # Log to wandb if enabled
                        if self.use_wandb:
                            try:
                                import wandb
                                wandb.log({f"train/loss_{name}": value}, step=trainer.global_step)
                            except Exception as e:
                                pass
            
            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                # Log component losses if they're available in the output
                if isinstance(outputs, dict) and 'loss_components' in outputs:
                    components = outputs['loss_components']
                    for name, value in components.items():
                        # Log to progress bar for important components
                        if name in ['nt_xent', 'recon', 'reg']:
                            trainer.logger.log_metrics({f"val/loss_{name}": value}, step=trainer.global_step)
                        
                        # Log to wandb if enabled
                        if self.use_wandb:
                            try:
                                import wandb
                                wandb.log({f"val/loss_{name}": value}, step=trainer.global_step)
                            except Exception as e:
                                pass
        
        # Add the detailed loss logger
        detailed_loss_logger = DetailedLossLogger(use_wandb=self.config.get('use_wandb', False))
        callbacks.append(detailed_loss_logger)
        
        # Configure logger
        loggers = []
        if self.config.get('use_wandb', False) and self.wandb_run is not None:
            try:
                import wandb
                from pytorch_lightning.loggers import WandbLogger
                
                # Create custom tags and config for better organization
                tags = [self.model_type, "texture", "contrastive"]
                if self.config.get('experiment_name'):
                    tags.append(self.config.get('experiment_name'))
                
                # Extract configuration for wandb
                wandb_config = {
                    "model_type": self.model_type,
                    "learning_rate": self.config.get('optimizer_config', {}).get('lr', 1e-4),
                    "weight_decay": self.config.get('optimizer_config', {}).get('weight_decay', 1e-4),
                    "batch_size": self.config.get('loader_config', {}).get('batch_size', 4),
                    "epochs": self.config.get('num_epochs', 50),
                    "architecture": "TextureNet_Lightning",
                    "loss_config": self.config.get('loss_config', {}),
                    "dataset": self.config.get('data_config', {}).get('root_dir', '')
                }
                
                # Set up logging of gradients and parameters
                wandb.watch(self.model, log="all", log_freq=10)
                
                # Create the wandb logger
                wandb_logger = WandbLogger(
                    project=self.config.get('wandb_project', 'MorphoFeatures'),
                    name=f"{self.model_type}_texture",
                    tags=tags,
                    config=wandb_config,
                    log_model=True,
                    save_dir=os.path.join(self.project_dir, 'logs')
                )
                
                # Add model graph visualization
                if hasattr(self.model, 'example_input_array'):
                    wandb_logger.watch(self.model, log_freq=100)
                    
                loggers.append(wandb_logger)
                logger.info(f"Initialized WandbLogger for {self.model_type} texture model")
            except Exception as e:
                logger.warning(f"Could not initialize WandbLogger: {e}")
        
        # Add TensorBoard logger
        from pytorch_lightning.loggers import TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(self.project_dir, 'logs'),
            name=f"{self.model_type}_texture"
        )
        loggers.append(tb_logger)
        
        # Add early stopping
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=True,
            mode='min'
        )
        callbacks.append(early_stop_callback)
        
        # Create the trainer
        self.trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=self.config.get('num_epochs', 50),
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=10,
            precision="16-mixed" if self.config.get('use_amp', False) else "32",
            default_root_dir=self.project_dir
        )
        
        logger.info("Created Lightning Trainer")
    
    def train(self):
        """Train the model using Lightning"""
        logger.info("Starting Lightning training")
        try:
            # Make sure trainer is created
            if self.trainer is None:
                self._setup_trainer()
                
            # Set up additional wandb logging for metrics that Lightning might not capture
            if self.config.get('use_wandb', False) and self.wandb_run is not None:
                # Create a callback to log after each epoch
                class WandbMetricsLogger(pl.Callback):
                    def on_train_epoch_end(self, trainer, pl_module):
                        if hasattr(trainer, 'callback_metrics'):
                            try:
                                import wandb
                                # Log all available metrics
                                metrics = {f"train/{k}": v for k, v in trainer.callback_metrics.items() 
                                          if not k.startswith('val_')}
                                wandb.log(metrics, step=trainer.global_step)
                            except Exception as e:
                                logger.warning(f"Wandb training metrics logging error: {e}")
                    
                    def on_validation_epoch_end(self, trainer, pl_module):
                        if hasattr(trainer, 'callback_metrics'):
                            try:
                                import wandb
                                # Log all available metrics
                                metrics = {f"val/{k}": v for k, v in trainer.callback_metrics.items() 
                                          if k.startswith('val_')}
                                wandb.log(metrics, step=trainer.global_step)
                                
                                # Log learning rate
                                if hasattr(trainer, 'optimizers') and trainer.optimizers:
                                    for i, optimizer in enumerate(trainer.optimizers):
                                        for param_group in optimizer.param_groups:
                                            wandb.log({f"lr/group_{i}": param_group['lr']}, 
                                                     step=trainer.global_step)
                            except Exception as e:
                                logger.warning(f"Wandb validation metrics logging error: {e}")
                
                # Add the callback to the trainer
                self.trainer.callbacks.append(WandbMetricsLogger())
                logger.info("Added wandb metrics logging callback to trainer")
                
            # Train the model
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
            
            # Log final best model path
            if self.config.get('use_wandb', False) and self.wandb_run is not None:
                try:
                    import wandb
                    best_model_path = self.trainer.checkpoint_callback.best_model_path
                    wandb.log({"best_model_path": best_model_path})
                    
                    # Log model as an artifact
                    artifact = wandb.Artifact(
                        name=f"{self.model_type}_texture_model", 
                        type="model",
                        description=f"Best {self.model_type} texture model"
                    )
                    artifact.add_file(best_model_path)
                    wandb.log_artifact(artifact)
                    
                    logger.info(f"Logged best model to wandb: {best_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to log best model to wandb: {e}")
            
            logger.info(f"Training completed. Best model saved at: {self.trainer.checkpoint_callback.best_model_path}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
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
    """
    Train a shape model using CustomShapeTrainer.
    
    Args:
        config_path (str): Path to YAML configuration file
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create the directory for experiments if it doesn't exist
    os.makedirs(config['experiment_dir'], exist_ok=True)
    
    # Configure wandb if requested
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'MorphoFeatures'),
            config=config
        )
        logger.info("Initialized wandb for logging")
    
    # Create the trainer
    logger.info("Creating CustomShapeTrainer")
    trainer = CustomShapeTrainer(config)
    
    # Train the model
    logger.info("Starting training")
    trainer.fit()
    
    return trainer

def train_texture_model(config_path, model_type='lowres'):
    """
    Train a texture model (low-resolution or high-resolution).
    
    Args:
        config_path (str): Path to YAML configuration file
        model_type (str): Type of texture model to train ('lowres' or 'highres')
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set model type in config
    config['model_type'] = model_type
    
    # Create the directory for experiments if it doesn't exist
    if 'project_directory' in config:
        os.makedirs(config['project_directory'], exist_ok=True)
    
    # Train a texture model using Lightning
    logger.info(f"Creating {model_type} TextureModelTrainer with Lightning")
    trainer = TextureModelTrainer(config, model_type=model_type)
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    return trainer

def train_texture_model_from_config(config, model_type='lowres'):
    """
    Train a texture model using TextureModelTrainer directly from a config dictionary.
    
    Args:
        config (dict): Configuration dictionary
        model_type (str): Type of texture model to train ('lowres' or 'highres')
    """
    # Set model type in config
    config['model_type'] = model_type
    
    # Create the directory for experiments if it doesn't exist
    if 'project_directory' in config:
        os.makedirs(config['project_directory'], exist_ok=True)
    
    # Configure wandb if requested
    wandb_run = None
    if config.get('use_wandb', False):
        try:
            import wandb
            if not wandb.api.api_key:
                # Try to login with the provided key or use environment variable
                wandb_key = os.environ.get('WANDB_API_KEY', "9de783cdb1f22a4b8f97f7e05e4e057f668e0cfe")
                wandb.login(key=wandb_key)
            
            # Define a more descriptive run name
            run_name = config.get('wandb_run_name', f"{model_type}_texture_model")
            if 'experiment_name' in config:
                run_name = f"{run_name}_{config['experiment_name']}"
                
            # Create tags for better organization
            tags = [model_type, "texture"]
            # Add any additional tags from config
            if config.get('tags'):
                tags.extend(config.get('tags'))
                
            # Add timestamp to run name for uniqueness
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            run_name = f"{run_name}_{timestamp}"
            
            # Start wandb run
            wandb_run = wandb.init(
                project=config.get('wandb_project', 'MorphoFeatures'),
                name=run_name,
                tags=tags,
                config=config,
                dir=os.environ.get("WANDB_DIR", None),
                group=config.get('wandb_group', None)
            )
            
            logger.info(f"Initialized wandb for {model_type} texture model logging: {run_name}")
            
            # Store the run in config for later access
            config['wandb_run'] = wandb_run
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb for texture model: {e}")
            config['use_wandb'] = False
    
    # Train a texture model using Lightning
    logger.info(f"Creating {model_type} TextureModelTrainer with Lightning")
    trainer = TextureModelTrainer(config, model_type=model_type)
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    
    # Finalize wandb if we started it
    if wandb_run is not None:
        try:
            # Log final metrics
            best_model_path = trainer.trainer.checkpoint_callback.best_model_path
            wandb_run.summary['best_model_path'] = best_model_path
            wandb_run.summary['best_val_loss'] = trainer.trainer.checkpoint_callback.best_model_score.item()
            wandb_run.summary['total_epochs'] = trainer.trainer.current_epoch
            
            # Add notes about the run
            notes = f"Model type: {model_type}\n"
            notes += f"Best validation loss: {trainer.trainer.checkpoint_callback.best_model_score.item():.6f}\n"
            notes += f"Trained for {trainer.trainer.current_epoch} epochs\n"
            wandb_run.notes = notes
            
            logger.info("Finalized wandb run with summary metrics")
        except Exception as e:
            logger.warning(f"Error finalizing wandb run: {e}")
    
    return trainer

def main():
    """Main entry point for training the models."""
    parser = argparse.ArgumentParser(description="Train MorphoFeatures models")
    # Original config-based arguments
    parser.add_argument("--shape_config", type=str, help="Path to shape model config")
    parser.add_argument("--lowres_config", type=str, help="Path to low-res texture model config")
    parser.add_argument("--highres_config", type=str, help="Path to high-res texture model config")
    parser.add_argument("--logging", type=str, default="INFO", help="Logging level")
    
    # New command-line arguments used in run_full_pipeline.sh
    parser.add_argument("--data-root", type=str, help="Root directory containing data samples")
    parser.add_argument("--low-res-dir", type=str, help="Directory with low-resolution data")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--model-type", type=str, choices=["shape", "nucleus"], help="Model type to train")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--gpu-id", type=int, help="GPU ID to use")
    parser.add_argument("--precomputed-dir", type=str, help="Directory for precomputed/cached meshes")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    
    # New loss function parameters
    parser.add_argument("--lambda-nt-xent", type=float, default=1.0, help="Weight for NT-Xent loss")
    parser.add_argument("--lambda-mse", type=float, default=0.5, help="Weight for MSE loss")
    parser.add_argument("--lambda-reg", type=float, default=0.1, help="Weight for regularization loss")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature parameter for NT-Xent loss")
    
    args = parser.parse_args()
    
    # Setup logging
    numeric_level = getattr(logging, args.logging.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.logging}")
    
    # Create a unique timestamp for this run
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Try to create wandb directory with a more robust approach
    try:
        # Use a timestamped wandb directory to avoid conflicts
        wandb_dir = f"wandb_runs/{timestamp}"
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        logger.info(f"Using wandb directory: {wandb_dir}")
    except OSError as e:
        logger.warning(f"Could not create wandb directory: {e}")
        logger.warning("Will use default wandb directory")
        # Let wandb handle directory creation
    
    logging.basicConfig(
        level=numeric_level,
        # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        # handlers=[
        #     logging.StreamHandler(),
        #     logging.FileHandler(f"morphofeatures_training_{timestamp}.log")
        # ]
    )
    
    # If command-line arguments are provided, create a config from them
    if args.data_root or args.model_type:
        # Check if we have all the required arguments
        if not (args.data_root and args.output_dir and args.model_type):
            logger.error("When using command-line arguments, --data-root, --output-dir, and --model-type are required.")
            parser.print_help()
            return 1
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if args.gpu_id is not None and not cuda_available:
            logger.warning("CUDA is not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"
        
        # Create a config based on command-line arguments
        if args.model_type == "shape":
            # Create shape model config
            config = {
                "experiment_dir": os.path.join(args.output_dir, "shape_model"),
                "device": device,
                "data": {
                    "root_dir": args.data_root,
                    "class_csv_path": "chromatin_classes_and_samples.csv",
                    "num_points": 1024,
                    "cache_dir": args.precomputed_dir,
                    "precomputed_dir": args.precomputed_dir
                },
                "loader": {
                    "batch_size": args.batch_size or 8,
                    "shuffle": True,
                    "num_workers": args.num_workers or 4
                },
                "model": {
                    "name": "DeepGCN",
                    "kwargs": {
                        "in_channels": 6,
                        "channels": 64,
                        "out_channels": 64,
                        "k": 12,
                        "norm": "batch",
                        "act": "relu",
                        "n_blocks": 14,
                        "projection_head": True,
                        "use_dilation": True
                    }
                },
                "optimizer": {
                    "name": "Adam",
                    "kwargs": {
                        "lr": 0.001,
                        "weight_decay": 0.0001
                    }
                },
                "criterion": {
                    "name": "ContrastiveLoss",
                    "kwargs": {
                        "pos_margin": 0,
                        "neg_margin": 1,
                        "distance": {
                            "function": "CosineSimilarity"
                        }
                    }
                },
                "training": {
                    "validate_every": 1,
                    "epochs": args.epochs or 50,
                    "checkpoint_every": 1
                },
                "scheduler": {
                    "step_size": 15,
                    "gamma": 0.5
                },
                "use_wandb": args.use_wandb,
                "wandb_project": "Chromatin",
                "wandb_run_name": "shape_model_training",
                "output": {
                    "checkpoint_dir": os.path.join(args.output_dir, "shape_model/checkpoints"),
                    "log_dir": os.path.join(args.output_dir, "shape_model/logs"),
                    "save_every": 5
                }
            }
            
            logger.info("Training shape model with generated config.")
            trainer = train_shape_model_from_config(config)
            logger.info("Shape model training completed.")
        
        elif args.model_type == "nucleus":
            # Create nucleus texture model config
            config = {
                "project_directory": os.path.join(args.output_dir, "nucleus_texture_model"),
                "device": device,
                "data_config": {
                    "root_dir": args.data_root,
                    "class_csv_path": os.path.join(args.data_root, "chromatin_classes_and_samples.csv"),
                    "box_size": (104, 104, 104),
                    "is_cytoplasm": False,
                    "input_dir": "raw",
                    "target_dir": "mask"
                },
                "loader_config": {
                    "batch_size": args.batch_size or 4,
                    "shuffle": True,
                    "num_workers": args.num_workers or 4
                },
                "val_loader_config": {
                    "batch_size": args.batch_size or 4,
                    "shuffle": False,
                    "num_workers": args.num_workers or 4
                },
                "model_config": {
                    "feature_dim": 64,
                    "in_channels": 1,
                    "out_channels": 1,
                    "f_maps": [32, 64, 128, 256]
                },
                "training_optimizer_kwargs": {
                    "optimizer_kwargs": {
                        "lr": 0.0001,
                        "weight_decay": 0.0001
                    }
                },
                "num_epochs": args.epochs or 50,
                "use_wandb": args.use_wandb,
                "wandb_project": "Chromatin",
                "wandb_run_name": "nucleus_texture_model_training",
                # Loss function parameters
                "lambda_nt_xent": args.lambda_nt_xent,
                "lambda_mse": args.lambda_mse,
                "lambda_reg": args.lambda_reg,
                "temperature": args.temperature
            }
            
            logger.info("Training nucleus texture model with generated config.")
            trainer = train_texture_model_from_config(config, model_type='highres')
            logger.info("Nucleus texture model training completed.")
    
    # Train models based on provided config files (original approach)
    else:
        try:
            if args.shape_config:
                logger.info(f"Training shape model with config: {args.shape_config}")
                trainer = train_shape_model(args.shape_config)
                logger.info("Shape model training completed.")
            
            if args.lowres_config:
                logger.info(f"Training low-res texture model with config: {args.lowres_config}")
                trainer = train_texture_model(args.lowres_config, model_type='lowres')
                logger.info("Low-res texture model training completed.")
            
            if args.highres_config:
                logger.info(f"Training high-res texture model with config: {args.highres_config}")
                trainer = train_texture_model(args.highres_config, model_type='highres')
                logger.info("High-res texture model training completed.")
            
            if not (args.shape_config or args.lowres_config or args.highres_config):
                logger.error("No configuration file provided. Please specify at least one model to train.")
                parser.print_help()
                return 1
        
        except Exception as e:
            logger.error(f"Error training models: {e}")
            traceback.print_exc()
            return 1
    
    return 0

def train_shape_model_from_config(config):
    """
    Train a shape model using CustomShapeTrainer directly from a config dictionary.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create the directory for experiments if it doesn't exist
    os.makedirs(config['experiment_dir'], exist_ok=True)
    
    # Configure wandb if requested
    if config.get('use_wandb', False):
        try:
            # Import wandb here to avoid early initialization
            import wandb
            # Only login if not already logged in
            if not wandb.api.api_key:
                wandb.login(key="9de783cdb1f22a4b8f97f7e05e4e057f668e0cfe")
            
            # Initialize wandb run with custom settings
            run = wandb.init(
                project=config.get('wandb_project', 'MorphoFeatures'),
                config=config,
                dir=os.environ.get("WANDB_DIR", None)  # Use environment variable if set
            )
            logger.info("Initialized wandb for logging")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            logger.warning("Continuing without wandb logging")
            config['use_wandb'] = False
    
    # Create the trainer
    logger.info("Creating CustomShapeTrainer")
    trainer = CustomShapeTrainer(config)
    
    # Train the model
    logger.info("Starting training")
    trainer.fit()
    
    return trainer

if __name__ == "__main__":
    sys.exit(main())