# adapted from Constantin Pape

import time
import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Import MONAI modules instead of inferno/neurofire
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter

# Fix the import path with try-except to handle different import configurations
try:
    # Try relative import first
    from .cell_loader import CellLoaders
except ImportError:
    try:
        # Try absolute import
        from morphofeatures.texture.cell_loader import CellLoaders
    except ImportError:
        # Fallback to direct import
        from cell_loader import CellLoaders

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_criterion(criterion, **criterion_kwargs):
    """Helper function to create loss criterion"""
    logger.info(f"Creating criterion from: {criterion} (type: {type(criterion)})")
    logger.info(f"Criterion kwargs: {criterion_kwargs}")
    
    # Handle integer criterion by converting to string (likely a reference to BCELoss)
    if isinstance(criterion, int):
        logger.warning(f"Received integer criterion: {criterion}, converting to 'BCELoss'")
        criterion = 'BCELoss'
    
    if isinstance(criterion, str):
        # Check standard PyTorch losses
        if hasattr(nn, criterion):
            logger.info(f"Using nn.{criterion}")
            pr_criterion = getattr(nn, criterion)(**criterion_kwargs)
        # Check MONAI losses
        elif hasattr(torch.nn, criterion):
            logger.info(f"Using torch.nn.{criterion}")
            pr_criterion = getattr(torch.nn, criterion)(**criterion_kwargs)
        # Try MONAI losses
        elif hasattr(DiceLoss, criterion):
            logger.info(f"Using DiceLoss.{criterion}")
            pr_criterion = getattr(DiceLoss, criterion)(**criterion_kwargs)
        else:
            error_msg = f"Unknown criterion: {criterion}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    elif isinstance(criterion, dict):
        # Multiple losses
        cr_list = []
        for cr in criterion:
            if hasattr(nn, cr):
                cr_list.append(getattr(nn, cr)(**criterion[cr]))
            elif hasattr(torch.nn, cr):
                cr_list.append(getattr(torch.nn, cr)(**criterion[cr]))
            elif hasattr(DiceLoss, cr):
                cr_list.append(getattr(DiceLoss, cr)(**criterion[cr]))
            else:
                raise ValueError(f"Unknown criterion: {cr}")
        
        # Create MultiLoss equivalent
        class MultiLoss(nn.Module):
            def __init__(self, criteria_list):
                super(MultiLoss, self).__init__()
                self.criteria = nn.ModuleList(criteria_list)
            
            def forward(self, input, target):
                loss = 0
                for criterion in self.criteria:
                    loss += criterion(input, target)
                return loss
        
        pr_criterion = MultiLoss(cr_list)
    else:
        pr_criterion = None
    return pr_criterion


def create_unet_model(config):
    """Create a MONAI UNet model based on the config"""
    model_kwargs = config.get('model_kwargs', {})
    
    # Set default parameters for MONAI UNet if not provided
    if 'in_channels' not in model_kwargs:
        model_kwargs['in_channels'] = 1
    if 'out_channels' not in model_kwargs:
        model_kwargs['out_channels'] = 1
    
    # Configure MONAI UNet with appropriate dimensions and channels
    spatial_dims = 3  # 3D UNet
    
    # Use smaller feature maps and fewer downsampling steps for small z-dimensions
    channels = model_kwargs.get('f_maps', [16, 32, 64, 128])
    
    # Limit depth of UNet based on input dimensions
    # For depth=8 after padding, we can have at most 3 downsampling operations
    # (8 -> 4 -> 2 -> 1)
    if len(channels) > 4:
        logger.info(f"Limiting UNet depth due to small z-dimension. Channels: {channels[:4]}")
        channels = channels[:4]  # Limit to 4 levels
    
    strides = [2] * (len(channels) - 1)
    
    logger.info(f"Creating UNet3D with channels: {channels}, strides: {strides}")
    
    # Create the MONAI UNet model with more explicit parameters
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=model_kwargs['in_channels'],
        out_channels=model_kwargs['out_channels'],
        channels=channels,
        strides=strides,
        num_res_units=2,
        norm='BATCH',
        dropout=0.0,
        kernel_size=3,
        up_kernel_size=3,
        act='PRELU',
    )
    
    # Add sigmoid activation if needed
    if model_kwargs.get('final_sigmoid', True):
        model = nn.Sequential(model, nn.Sigmoid())
    
    return model


class ModelTrainer:
    """MONAI-based trainer to replace inferno Trainer"""
    def __init__(self, model):
        self.model = model
        self.criterion = None
        self.validation_criterion = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.save_dir = None
        self.best_val_loss = float('inf')
        self.iterations = 0
        self.validate_every = 100
        self.validate_for = 20
        self.current_epoch = 0
        self.max_epochs = 10
        self.smoothness = 0.95
        self.validation_history = []
        self.scheduler = None
        
    def build_criterion(self, criterion):
        self.criterion = criterion
        return self
    
    def build_validation_criterion(self, criterion):
        self.validation_criterion = criterion
        return self
    
    def build_optimizer(self, optimizer='Adam', **optimizer_kwargs):
        optimizer_class = getattr(torch.optim, optimizer)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        return self
    
    def set_validate_every(self, validate_every, for_num_iterations=20):
        """
        Set validation interval
        
        Args:
            validate_every: Can be an int or a tuple (iterations, 'iterations')
            for_num_iterations: Number of iterations to validate for
        """
        logger.info(f"Setting validate_every with value: {validate_every} (type: {type(validate_every)})")
        
        if isinstance(validate_every, tuple):
            # Handle tuple case (iterations, 'iterations')
            try:
                self.validate_every = validate_every[0]
                logger.info(f"Set self.validate_every = {self.validate_every} from tuple[0]")
            except (IndexError, TypeError) as e:
                logger.error(f"Error extracting value from validate_every tuple: {e}")
                # Use default as fallback
                self.validate_every = 100
                logger.info(f"Using default validate_every = {self.validate_every}")
        else:
            # Handle integer or other types
            try:
                self.validate_every = int(validate_every)
                logger.info(f"Set self.validate_every = {self.validate_every} from direct value")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting validate_every to int: {e}")
                # Use default as fallback
                self.validate_every = 100
                logger.info(f"Using default validate_every = {self.validate_every}")
                
        self.validate_for = for_num_iterations
        return self
    
    def save_every(self, save_every, to_directory=None):
        """
        Set checkpoint saving interval
        
        Args:
            save_every: Can be an int or a tuple (iterations, 'iterations')
            to_directory: Directory to save checkpoints
        """
        logger.info(f"Setting save_every with value: {save_every} (type: {type(save_every)})")
        
        if isinstance(save_every, tuple):
            # Handle tuple case (iterations, 'iterations')
            try:
                self.save_every_n = save_every[0]
                logger.info(f"Set self.save_every_n = {self.save_every_n} from tuple[0]")
            except (IndexError, TypeError) as e:
                logger.error(f"Error extracting value from save_every tuple: {e}")
                # Use default as fallback
                self.save_every_n = 1000
                logger.info(f"Using default save_every_n = {self.save_every_n}")
        else:
            # Handle integer or other types
            try:
                self.save_every_n = int(save_every)
                logger.info(f"Set self.save_every_n = {self.save_every_n} from direct value")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting save_every to int: {e}")
                # Use default as fallback
                self.save_every_n = 1000
                logger.info(f"Using default save_every_n = {self.save_every_n}")
        
        self.save_dir = to_directory
        if to_directory:
            os.makedirs(self.save_dir, exist_ok=True)
        return self
    
    def set_max_num_epochs(self, epochs):
        self.max_epochs = epochs
        return self
    
    def bind_loader(self, name, loader):
        if name == 'train':
            self.train_loader = loader
        elif name == 'validate':
            self.val_loader = loader
        return self
    
    def build_logger(self, logger_type=None, log_directory=None):
        if log_directory is not None:
            os.makedirs(log_directory, exist_ok=True)
            self.writer = SummaryWriter(log_directory)
        return self
    
    def register_callback(self, callback):
        """Support for AutoLR scheduler and other callbacks"""
        if hasattr(callback, 'factor') and hasattr(callback, 'patience'):
            # This is likely an AutoLR equivalent, create ReduceLROnPlateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', 
                factor=callback.factor,
                patience=int(callback.patience.split()[0]) if isinstance(callback.patience, str) else callback.patience,
                verbose=True
            )
        return self
    
    def fit(self):
        """Main training loop"""
        self.model.to(self.device)
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            val_loss = self._validate_epoch()
            
            # Save checkpoint periodically
            if epoch % 5 == 0 or epoch == self.max_epochs - 1:
                self._save_checkpoint()
            
            # Update scheduler if available
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                
            logger.info(f"Epoch {epoch} completed. Validation loss: {val_loss:.6f}")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")):
            # Get the inputs and targets
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                if isinstance(targets, (list, tuple)):
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs  # For autoencoder-like models
            
            # Handle contrastive data format if needed (reshape tensors)
            if len(inputs.shape) > 5:  # If contrastive data format with extra dimensions
                # Try to adapt the shape - reshape to standard 5D format
                # For contrastive format [B, 2, C, D, H, W] reshape to [B*2, C, D, H, W]
                if inputs.shape[1] == 2:  # Check if it has contrastive pairs dimension
                    batch_size = inputs.shape[0]
                    inputs = inputs.view(batch_size * 2, *inputs.shape[2:])
                    if not isinstance(targets, list) and len(targets.shape) > 4:
                        targets = targets.view(batch_size * 2, *targets.shape[2:])
                # For other unusual formats, try to infer the correct reshape
                elif len(inputs.shape) == 6:
                    logger.info(f"Unusual input shape: {inputs.shape}, attempting reshape")
                    # Try to reshape to [B, C, D, H, W]
                    inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[3], 
                                        inputs.shape[4], inputs.shape[5])
            
            # Special handling for 3D inputs with small depth dimension
            # MONAI UNet may have issues with very small depth dimensions (like 5)
            # We'll add padding to ensure depth is at least 8 (power of 2)
            if len(inputs.shape) == 5 and inputs.shape[2] < 8:
                logger.info(f"Small depth dimension detected: {inputs.shape[2]}, padding to 8")
                # Calculate padding needed
                pad_size = 8 - inputs.shape[2]
                padding = (0, 0, 0, 0, 0, pad_size)  # Padding format: (left, right, top, bottom, front, back)
                # Apply padding
                inputs = torch.nn.functional.pad(inputs, padding, mode='constant', value=0)
                if not isinstance(targets, list) and len(targets.shape) == 5:
                    targets = torch.nn.functional.pad(targets, padding, mode='constant', value=0)
            
            # Normalize target to [0,1] range for BCELoss
            if not isinstance(targets, list):
                # Convert to float if needed
                if not torch.is_floating_point(targets):
                    logger.info(f"Converting target dtype from {targets.dtype} to float")
                    targets = targets.float()
                
                # Check if normalization is needed
                if targets.min() < 0 or targets.max() > 1:
                    logger.info(f"Normalizing targets from range [{targets.min().item():.4f}, {targets.max().item():.4f}] to [0,1]")
                    # Apply min-max normalization
                    if targets.max() > targets.min():  # Avoid division by zero
                        targets = (targets - targets.min()) / (targets.max() - targets.min())
                    else:
                        # If all values are the same, set to either 0 or 1 based on value
                        targets = (targets > 0).float()
            
            # Log the actual shapes after reshape for debugging
            logger.info(f"Input shape after processing: {inputs.shape}")
            if not isinstance(targets, list):
                logger.info(f"Target shape after processing: {targets.shape}")
                # Log target value range
                logger.info(f"Target value range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward + optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            
            # Log to tensorboard
            if self.writer is not None and batch_idx % 10 == 0:
                self.writer.add_scalar('training/loss', loss.item(), self.iterations)
            
            # Validate periodically
            self.iterations += 1
            if self.iterations % self.validate_every == 0:
                self._validate_iteration()
        
        epoch_loss = running_loss / len(self.train_loader)
        logger.info(f"Epoch {self.current_epoch} - Training loss: {epoch_loss:.6f}")
        
        return epoch_loss
    
    def _validate_iteration(self):
        """Run validation for a few iterations"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.validate_for:
                    break
                
                # Get the inputs and targets
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    if isinstance(targets, (list, tuple)):
                        targets = [t.to(self.device) for t in targets]
                    else:
                        targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # For autoencoder-like models
                
                # Handle contrastive data format if needed (reshape tensors)
                if len(inputs.shape) > 5:  # If contrastive data format with extra dimensions
                    # Try to adapt the shape - reshape to standard 5D format
                    # For contrastive format [B, 2, C, D, H, W] reshape to [B*2, C, D, H, W]
                    if inputs.shape[1] == 2:  # Check if it has contrastive pairs dimension
                        batch_size = inputs.shape[0]
                        inputs = inputs.view(batch_size * 2, *inputs.shape[2:])
                        if not isinstance(targets, list) and len(targets.shape) > 4:
                            targets = targets.view(batch_size * 2, *targets.shape[2:])
                    # For other unusual formats, try to infer the correct reshape
                    elif len(inputs.shape) == 6:
                        logger.info(f"Unusual input shape: {inputs.shape}, attempting reshape")
                        # Try to reshape to [B, C, D, H, W]
                        inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[3], 
                                            inputs.shape[4], inputs.shape[5])
                
                # Special handling for 3D inputs with small depth dimension
                # MONAI UNet may have issues with very small depth dimensions (like 5)
                # We'll add padding to ensure depth is at least 8 (power of 2)
                if len(inputs.shape) == 5 and inputs.shape[2] < 8:
                    logger.info(f"Small depth dimension detected: {inputs.shape[2]}, padding to 8")
                    # Calculate padding needed
                    pad_size = 8 - inputs.shape[2]
                    padding = (0, 0, 0, 0, 0, pad_size)  # Padding format: (left, right, top, bottom, front, back)
                    # Apply padding
                    inputs = torch.nn.functional.pad(inputs, padding, mode='constant', value=0)
                    if not isinstance(targets, list) and len(targets.shape) == 5:
                        targets = torch.nn.functional.pad(targets, padding, mode='constant', value=0)
                
                # Normalize target to [0,1] range for BCELoss
                if not isinstance(targets, list):
                    # Convert to float if needed
                    if not torch.is_floating_point(targets):
                        logger.info(f"Converting target dtype from {targets.dtype} to float")
                        targets = targets.float()
                    
                    # Check if normalization is needed
                    if targets.min() < 0 or targets.max() > 1:
                        logger.info(f"Normalizing validation targets from range [{targets.min().item():.4f}, {targets.max().item():.4f}] to [0,1]")
                        # Apply min-max normalization
                        if targets.max() > targets.min():  # Avoid division by zero
                            targets = (targets - targets.min()) / (targets.max() - targets.min())
                        else:
                            # If all values are the same, set to either 0 or 1 based on value
                            targets = (targets > 0).float()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.validation_criterion(outputs, targets)
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        
        # Log to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('validation/loss', val_loss, self.iterations)
        
        # Update validation history with smoothing
        if not self.validation_history:
            self.validation_history.append(val_loss)
        else:
            smoothed_loss = (self.smoothness * self.validation_history[-1] + 
                            (1 - self.smoothness) * val_loss)
            self.validation_history.append(smoothed_loss)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint(is_best=True)
            logger.info(f"New best validation loss: {val_loss:.6f}")
        
        self.model.train()
        return val_loss
    
    def _validate_epoch(self):
        """Run validation on the entire validation set"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Get the inputs and targets
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    if isinstance(targets, (list, tuple)):
                        targets = [t.to(self.device) for t in targets]
                    else:
                        targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # For autoencoder-like models
                
                # Handle contrastive data format if needed (reshape tensors)
                if len(inputs.shape) > 5:  # If contrastive data format with extra dimensions
                    # Try to adapt the shape - reshape to standard 5D format
                    # For contrastive format [B, 2, C, D, H, W] reshape to [B*2, C, D, H, W]
                    if inputs.shape[1] == 2:  # Check if it has contrastive pairs dimension
                        batch_size = inputs.shape[0]
                        inputs = inputs.view(batch_size * 2, *inputs.shape[2:])
                        if not isinstance(targets, list) and len(targets.shape) > 4:
                            targets = targets.view(batch_size * 2, *targets.shape[2:])
                    # For other unusual formats, try to infer the correct reshape
                    elif len(inputs.shape) == 6:
                        logger.info(f"Unusual input shape: {inputs.shape}, attempting reshape")
                        # Try to reshape to [B, C, D, H, W]
                        inputs = inputs.view(inputs.shape[0], inputs.shape[2], inputs.shape[3], 
                                            inputs.shape[4], inputs.shape[5])
                
                # Special handling for 3D inputs with small depth dimension
                # MONAI UNet may have issues with very small depth dimensions (like 5)
                # We'll add padding to ensure depth is at least 8 (power of 2)
                if len(inputs.shape) == 5 and inputs.shape[2] < 8:
                    logger.info(f"Small depth dimension detected: {inputs.shape[2]}, padding to 8")
                    # Calculate padding needed
                    pad_size = 8 - inputs.shape[2]
                    padding = (0, 0, 0, 0, 0, pad_size)  # Padding format: (left, right, top, bottom, front, back)
                    # Apply padding
                    inputs = torch.nn.functional.pad(inputs, padding, mode='constant', value=0)
                    if not isinstance(targets, list) and len(targets.shape) == 5:
                        targets = torch.nn.functional.pad(targets, padding, mode='constant', value=0)
                
                # Normalize target to [0,1] range for BCELoss
                if not isinstance(targets, list):
                    # Convert to float if needed
                    if not torch.is_floating_point(targets):
                        logger.info(f"Converting target dtype from {targets.dtype} to float")
                        targets = targets.float()
                    
                    # Check if normalization is needed
                    if targets.min() < 0 or targets.max() > 1:
                        logger.info(f"Normalizing validation targets from range [{targets.min().item():.4f}, {targets.max().item():.4f}] to [0,1]")
                        # Apply min-max normalization
                        if targets.max() > targets.min():  # Avoid division by zero
                            targets = (targets - targets.min()) / (targets.max() - targets.min())
                        else:
                            # If all values are the same, set to either 0 or 1 based on value
                            targets = (targets > 0).float()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.validation_criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        
        # Log to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('validation/epoch_loss', val_loss, self.current_epoch)
        
        return val_loss
    
    def _save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        if self.save_dir is None:
            return
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'iterations': self.iterations
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pytorch')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if requested
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pytorch')
            torch.save(checkpoint, best_path)
    
    def load(self, from_directory=None, filename='checkpoint.pytorch', best=False):
        """Load a saved model"""
        if best:
            filename = 'best_model.pytorch'
        
        checkpoint_path = os.path.join(from_directory, filename)
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.iterations = checkpoint.get('iterations', 0)
        
        return self
    
    def cuda(self, device_ids=None):
        """Move model to CUDA"""
        if device_ids is None:
            self.model = self.model.to(self.device)
        else:
            if not isinstance(self.model, torch.nn.DataParallel):
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.to(self.device)
        return self
    
    def cpu(self):
        """Move model to CPU"""
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        return self


def set_up_training(project_directory, config):
    """Set up training with MONAI UNet instead of neurofire models"""
    # Create model based on config
    model_name = config.get('model_name')
    
    # Use MONAI UNet for standard model types
    if model_name == 'UNet3D':
        model = create_unet_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Please use 'UNet3D' or extend the implementation.")
    
    # Compile criterion
    criterion = compile_criterion(config.get('loss'), **config.get('loss_kwargs', {}))
    
    logger.info("Building trainer.")
    smoothness = config.get('smoothness', 0.95)
    
    # Create trainer
    trainer = ModelTrainer(model)
    trainer.smoothness = smoothness
    trainer.build_criterion(criterion)
    trainer.build_validation_criterion(criterion)
    
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
    trainer.set_validate_every((100, 'iterations'), for_num_iterations=20)
    trainer.save_every((1000, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))
    
    # Create AutoLR equivalent
    class AutoLR:
        def __init__(self, factor, patience, monitor, monitor_while, monitor_momentum, consider_improvement_with_respect_to):
            self.factor = factor
            self.patience = patience
            self.monitor = monitor
            self.monitor_while = monitor_while
            self.monitor_momentum = monitor_momentum
            self.consider_improvement_with_respect_to = consider_improvement_with_respect_to
    
    auto_lr = AutoLR(
        factor=0.98,
        patience='100 iterations',
        monitor='validation_loss_averaged',
        monitor_while='validating',
        monitor_momentum=smoothness,
        consider_improvement_with_respect_to='previous'
    )
    
    trainer.register_callback(auto_lr)
    
    # Set up tensorboard logging
    logger.info("Building logger.")
    trainer.build_logger(log_directory=os.path.join(project_directory, 'Logs'))
    
    return trainer


def yaml2dict(yaml_path):
    """Load YAML as dictionary - replacing inferno's yaml2dict"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def training(project_directory, train_configuration_file,
             data_configuration_file, devices, from_checkpoint):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)
    logger.info("Using devices {}".format(devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    if from_checkpoint:
        try:
            # Create a basic trainer to load the model
            temp_model = nn.Sequential(UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[32, 64, 128, 256]), nn.Sigmoid())
            trainer = ModelTrainer(temp_model)
            trainer = trainer.load(from_directory=os.path.join(project_directory, 'Weights'),
                                filename='checkpoint.pytorch')
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    else:
        trainer = set_up_training(project_directory, config)

    logger.info("Loading training and validation data loader from %s." % data_configuration_file)
    loader = CellLoaders(data_configuration_file)
    train_loader, validation_loader = loader.get_train_loaders()
    trainer.set_max_num_epochs(config.get('num_epochs', 10))

    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader('validate', validation_loader)

    # Move to CUDA if available
    if torch.cuda.is_available():
        device_ids = [i for i in range(len(devices.split(',')))]
        if len(device_ids) > 0:
            trainer.cuda(device_ids=device_ids if len(device_ids) > 1 else None)
    
    logger.info("Lift off!")
    start = time.time()
    trainer.fit()
    end = time.time()
    time_diff = end - start
    print("The training took {0} hours {1} minutes".format(time_diff // 3600,
                                                           time_diff % 3600 // 60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('--devices', type=str, default='2')
    parser.add_argument('--from_checkpoint', type=int, default=0)

    args = parser.parse_args()

    project_directory = args.project_directory
    assert os.path.exists(project_directory), 'create a project directory with config files!'

    train_config = os.path.join(project_directory, 'train_config.yml')
    data_config = os.path.join(project_directory, 'data_config.yml')

    training(project_directory, train_config, data_config,
             devices=args.devices, from_checkpoint=args.from_checkpoint)


if __name__ == '__main__':
    main()
