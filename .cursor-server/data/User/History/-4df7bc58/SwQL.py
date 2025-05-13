#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
import time
import sys
from tqdm import tqdm
import monai
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImage, AddChannel, ScaleIntensity, ToTensor, Compose,
    RandRotate90, RandFlip, RandZoom, RandGaussianNoise
)

# Setup logging
logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s',
                   stream=sys.stdout,
                   level=logging.INFO)
logger = logging.getLogger(__name__)

class TextureDataset(Dataset):
    """
    Custom dataset for loading 3D texture data
    """
    def __init__(self, root_dir, sample_ids, class_csv_path=None, 
                 is_cytoplasm=False, box_size=(104, 104, 104), debug=False):
        self.root_dir = root_dir
        self.sample_ids = sample_ids
        self.is_cytoplasm = is_cytoplasm
        self.box_size = box_size
        self.debug = debug
        
        # Load class information if available
        if class_csv_path:
            df = pd.read_csv(class_csv_path)
            self.sample_to_class = dict(zip(df['sample_id'].astype(str), df['class_id']))
        else:
            self.sample_to_class = None
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Define paths for raw and mask files
        raw_file = os.path.join(self.root_dir, sample_id, 'raw.npy')
        mask_file = os.path.join(self.root_dir, sample_id, 
                                'cytoplasm_mask.npy' if self.is_cytoplasm else 'nuclear_mask.npy')
        
        # Load data
        try:
            raw_data = np.load(raw_file).astype(np.float32)
            mask_data = np.load(mask_file).astype(np.float32)
            
            # Normalize raw data to [0, 1]
            if np.max(raw_data) > 0:
                raw_data = raw_data / np.max(raw_data)
            
            # Crop or pad to box_size
            raw_cropped = self._crop_or_pad(raw_data, self.box_size)
            mask_cropped = self._crop_or_pad(mask_data, self.box_size)
            
            # Add channel dimension
            raw_tensor = torch.from_numpy(raw_cropped[np.newaxis, ...])
            mask_tensor = torch.from_numpy(mask_cropped[np.newaxis, ...])
            
            # Return with sample ID
            return {
                'input': raw_tensor,
                'target': mask_tensor,
                'sample_id': sample_id
            }
        except Exception as e:
            if self.debug:
                print(f"Error loading sample {sample_id}: {e}")
            # Return a dummy sample
            dummy_input = torch.zeros((1, *self.box_size), dtype=torch.float32)
            dummy_target = torch.zeros((1, *self.box_size), dtype=torch.float32)
            return {'input': dummy_input, 'target': dummy_target, 'sample_id': sample_id}
    
    def _crop_or_pad(self, data, target_shape):
        """Crop or pad data to target shape"""
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Compute dimensions to copy
        copy_shape = [min(s, t) for s, t in zip(data.shape, target_shape)]
        
        # Compute slices for source and destination
        src_slices = tuple(slice(0, s) for s in copy_shape)
        dst_slices = tuple(slice(0, s) for s in copy_shape)
        
        # Copy data
        result[dst_slices] = data[src_slices]
        
        return result


def get_morphofeatures_texture_dataloader(root_dir, batch_size=4, shuffle=True, num_workers=4,
                                         class_csv_path=None, sample_ids=None, is_cytoplasm=False,
                                         box_size=(104, 104, 104), pin_memory=True, debug=False):
    """
    Create a DataLoader for texture data
    """
    # If sample_ids not provided, get all directories in root_dir
    if sample_ids is None:
        sample_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Create dataset
    dataset = TextureDataset(
        root_dir=root_dir,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        is_cytoplasm=is_cytoplasm,
        box_size=box_size,
        debug=debug
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


class PyTorchTextureTrainer:
    """
    Trainer for texture models using MONAI/PyTorch
    """
    def __init__(self, config, model_type='lowres'):
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Setup directories
        self.project_dir = config.get('project_directory', f'experiments/{model_type}_texture_model')
        os.makedirs(os.path.join(self.project_dir, 'Weights'), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, 'Logs'), exist_ok=True)
        
        # Setup the model
        self.setup_model()
        
        # Setup dataloaders
        self.setup_dataloaders()
        
        # Initialize best validation loss
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.num_epochs = config.get('num_epochs', 50)
    
    def setup_model(self):
        """Set up the MONAI UNet model, criterion, and optimizer"""
        # Get model configuration
        model_kwargs = self.config.get('model_kwargs', {})
        
        # Set default parameters for MONAI UNet if not provided
        if 'in_channels' not in model_kwargs:
            model_kwargs['in_channels'] = 1
        if 'out_channels' not in model_kwargs:
            model_kwargs['out_channels'] = 1
        
        # Configure MONAI UNet with appropriate dimensions and channels
        dimensions = 3  # 3D UNet
        channels = model_kwargs.get('f_maps', [32, 64, 128, 256])
        strides = [2] * (len(channels) - 1)
        
        # Create the MONAI UNet model
        self.model = UNet(
            dimensions=dimensions,
            in_channels=model_kwargs['in_channels'],
            out_channels=model_kwargs['out_channels'],
            channels=channels,
            strides=strides,
            num_res_units=2
        )
        
        # Add sigmoid activation if needed
        if model_kwargs.get('final_sigmoid', True):
            self.model = nn.Sequential(self.model, nn.Sigmoid())
        
        self.model.to(self.device)
        
        # Setup criterion
        criterion_name = self.config.get('loss', 'BCELoss')
        criterion_kwargs = self.config.get('loss_kwargs', {})
        
        if hasattr(nn, criterion_name):
            self.criterion = getattr(nn, criterion_name)(**criterion_kwargs)
        elif hasattr(monai.losses, criterion_name):
            self.criterion = getattr(monai.losses, criterion_name)(**criterion_kwargs)
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        
        # Setup optimizer
        optimizer_config = self.config.get('training_optimizer_kwargs', {})
        optimizer_name = optimizer_config.get('optimizer', 'Adam')
        optimizer_kwargs = optimizer_config.get('optimizer_kwargs', {})
        
        if hasattr(optim, optimizer_name):
            self.optimizer = getattr(optim, optimizer_name)(
                self.model.parameters(), **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    def setup_dataloaders(self):
        """Setup dataloaders for training and validation"""
        print(f"Setting up {self.model_type} texture dataloaders")
        
        # Get data configuration
        data_config = self.config.get('data_config', {})
        root_dir = data_config.get('root_dir', 'data')
        class_csv_path = data_config.get('class_csv_path', None)
        is_cytoplasm = data_config.get('is_cytoplasm', False)
        box_size = data_config.get('box_size', (104, 104, 104))
        split = data_config.get('split', 0.2)
        seed = data_config.get('seed', 42)
        
        # Get loader configs
        train_loader_config = self.config.get('loader_config', {})
        val_loader_config = self.config.get('val_loader_config', {})
        
        # Load sample IDs from CSV
        if class_csv_path and os.path.exists(class_csv_path):
            df = pd.read_csv(class_csv_path)
            sample_ids = df['sample_id'].astype(str).tolist()
            
            # Split the data
            np.random.seed(seed)
            np.random.shuffle(sample_ids)
            split_idx = int(len(sample_ids) * (1 - split))
            train_sample_ids = sample_ids[:split_idx]
            val_sample_ids = sample_ids[split_idx:]
            
            print(f"Total samples: {len(sample_ids)}")
            print(f"Training samples: {len(train_sample_ids)}")
            print(f"Validation samples: {len(val_sample_ids)}")
            
            # Create training dataloader
            self.train_loader = get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                batch_size=train_loader_config.get('batch_size', 4),
                shuffle=train_loader_config.get('shuffle', True),
                num_workers=train_loader_config.get('num_workers', 4),
                class_csv_path=class_csv_path,
                sample_ids=train_sample_ids,
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                pin_memory=train_loader_config.get('pin_memory', True)
            )
            
            # Create validation dataloader
            self.val_loader = get_morphofeatures_texture_dataloader(
                root_dir=root_dir,
                batch_size=val_loader_config.get('batch_size', 4),
                shuffle=False,
                num_workers=val_loader_config.get('num_workers', 4),
                class_csv_path=class_csv_path,
                sample_ids=val_sample_ids,
                is_cytoplasm=is_cytoplasm,
                box_size=box_size,
                pin_memory=val_loader_config.get('pin_memory', True)
            )
        else:
            raise ValueError("Class CSV file not found. A valid class CSV file is required to create dataloaders.")
    
    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")):
            # Get the inputs and targets
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward + optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            
            # Print statistics periodically
            if batch_idx % 10 == 9:  # print every 10 batches
                logger.info(f'[{self.current_epoch + 1}, {batch_idx + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        
        return running_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Get the inputs and targets
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track statistics
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        logger.info(f'Validation Loss: {avg_val_loss:.3f}')
        
        # Check if this is the best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(is_best=True)
            logger.info(f'New best validation loss: {avg_val_loss:.3f}')
        
        # Update the scheduler
        self.scheduler.step(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, is_best=False):
        """Save the model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save the checkpoint
        checkpoint_path = os.path.join(self.project_dir, 'Weights', f'checkpoint_epoch_{self.current_epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, create a separate checkpoint
        if is_best:
            best_path = os.path.join(self.project_dir, 'Weights', 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model to {best_path}')
    
    def train(self):
        """Train the model for the specified number of epochs"""
        logger.info(f"Starting {self.model_type} texture model training with MONAI")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"The {self.model_type} texture training took {training_time // 3600} hours {training_time % 3600 // 60} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.3f}")
        
        return self.best_val_loss 