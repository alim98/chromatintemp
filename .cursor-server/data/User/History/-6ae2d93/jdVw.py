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

from cell_loader import CellLoaders

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_criterion(criterion, **criterion_kwargs):
    """Helper function to create loss criterion"""
    if isinstance(criterion, str):
        # Check standard PyTorch losses
        if hasattr(nn, criterion):
            pr_criterion = getattr(nn, criterion)(**criterion_kwargs)
        # Check MONAI losses
        elif hasattr(torch.nn, criterion):
            pr_criterion = getattr(torch.nn, criterion)(**criterion_kwargs)
        # Try MONAI losses
        elif hasattr(DiceLoss, criterion):
            pr_criterion = getattr(DiceLoss, criterion)(**criterion_kwargs)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
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
    dimensions = 3  # 3D UNet
    channels = model_kwargs.get('f_maps', [32, 64, 128, 256])
    strides = [2] * (len(channels) - 1)
    
    # Create the MONAI UNet model
    model = UNet(
        dimensions=dimensions,
        in_channels=model_kwargs['in_channels'],
        out_channels=model_kwargs['out_channels'],
        channels=channels,
        strides=strides,
        num_res_units=2
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
    
    def validate_every(self, validate_every, for_num_iterations=20):
        self.validate_every = validate_every[0] if isinstance(validate_every, tuple) else validate_every
        self.validate_for = for_num_iterations
        return self
    
    def save_every(self, save_every, to_directory=None):
        self.save_every_n = save_every[0] if isinstance(save_every, tuple) else save_every
        self.save_dir = to_directory
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
    trainer.build_optimizer(**config.get('training_optimizer_kwargs', {}))
    trainer.validate_every((100, 'iterations'), for_num_iterations=20)
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
        # Create a basic trainer to load the model
        temp_model = nn.Sequential(UNet(dimensions=3, in_channels=1, out_channels=1, channels=[32, 64, 128, 256]), nn.Sigmoid())
        trainer = ModelTrainer(temp_model)
        trainer = trainer.load(from_directory=os.path.join(project_directory, 'Weights'),
                               filename='checkpoint.pytorch')
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
