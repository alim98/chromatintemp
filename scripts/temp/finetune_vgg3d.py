import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.cuda.amp as amp
from torch.utils.data import random_split
import matplotlib.cm as cm
import seaborn as sns
import time
import threading
import psutil
import gc
from functools import partial
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader
from model.vgg3d import Vgg3D, load_model_from_checkpoint

# Custom adapter model to handle size mismatches
class Vgg3DAdapter(nn.Module):
    def __init__(self, base_model, input_size, num_classes, classifier_size=512):
        super(Vgg3DAdapter, self).__init__()
        self.features = base_model.features
        
        # Calculate the feature output size with a forward pass
        test_input = torch.zeros(1, 1, *input_size)
        with torch.no_grad():
            feature_output = self.features(test_input)
        feature_size = feature_output.view(1, -1).size(1)
        
        print(f"Feature extractor produces {feature_size} features")
        
        # Create a new classifier that matches the feature size
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_size, classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_size, num_classes),
        )
        
        # Initialize the classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune VGG3D model on nuclei dataset')
    
    # Data params
    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT, 
                        help='Path to nuclei dataset')
    parser.add_argument('--class_csv', type=str, default=config.CLASS_CSV_PATH,
                        help='Path to class CSV file')
    parser.add_argument('--index_csv', type=str, default=os.path.join(config.ANALYSIS_OUTPUT_DIR, 'nuclei_index.csv'),
                        help='Path to CSV index file (for efficient data loading)')
    parser.add_argument('--create_index', action='store_true',
                        help='Create a new index file even if one exists')
    parser.add_argument('--output_dir', type=str, default=config.VISUALIZATION_OUTPUT_DIR,
                        help='Directory to save results and model checkpoints')
    parser.add_argument('--class_id', type=int, nargs='+', default=None,
                        help='Filter by class ID(s). If not specified, all classes will be included.')
    parser.add_argument('--subsample_ratio', type=float, default=config.DATA_SUBSAMPLE_RATIO,
                        help='Ratio of data to use (0.0-1.0). Default is set in config.py')
    parser.add_argument('--sample_percent', type=int, default=100,
                        help='Percentage of samples to load per class (1-100)')
    
    # Model params
    parser.add_argument('--checkpoint', type=str, default='model/hemibrain_production.checkpoint',
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--target_size', type=int, nargs=3, default=[80, 80, 80],
                        help='Target size for volumes (depth, height, width)')
    parser.add_argument('--output_classes', type=int, default=None,
                        help='Number of output classes. If not specified, will be determined from the dataset')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 regularization) to reduce overfitting')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam',
                        help='Optimizer to use for training')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--freeze_features', action='store_true', default=True,
                        help='Freeze feature extractor layers and only train classifier (default: True)')
    parser.add_argument('--freeze_classifier_layers', type=int, default=0,
                        help='Number of classifier layers to freeze (0-5, 0 means train all classifier layers)')
    parser.add_argument('--reduce_classifier', action='store_true', default=False,
                        help='Reduce the size of classifier layers to decrease trainable parameters')
    parser.add_argument('--classifier_size', type=int, default=512,
                        help='Size of the reduced classifier layer (only used if --reduce_classifier is set)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training (vs validation)')
    
    # New training enhancements
    parser.add_argument('--scheduler', type=str, choices=['reduce_on_plateau', 'cosine', 'none'], default='cosine',
                        help='Learning rate scheduler to use')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='Use mixed precision training (faster but may affect accuracy)')
    parser.add_argument('--progressive_unfreezing', action='store_true',
                        help='Gradually unfreeze layers during training for better transfer learning')
    parser.add_argument('--validation_freq', type=int, default=1,
                        help='Validate every N epochs (default: end of each epoch)')
    parser.add_argument('--save_best_metrics', type=str, nargs='+', default=['loss', 'accuracy'],
                        choices=['loss', 'accuracy', 'f1'], help='Metrics to use for saving best models')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation during training')
    
    # Debugging and monitoring
    parser.add_argument('--memory_monitor', action='store_true',
                        help='Enable memory monitoring during training')
    parser.add_argument('--monitor_interval', type=float, default=10.0,
                        help='Interval in seconds between memory monitoring reports')
    parser.add_argument('--cuda_launch_blocking', action='store_true',
                        help='Set CUDA_LAUNCH_BLOCKING=1 for better error reporting')
    parser.add_argument('--force_cpu_validation', action='store_true',
                        help='Force validation to be run on CPU to avoid CUDA errors')
    
    # Training process
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers for data loading (0 for main process only)')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory for data loading (may cause CUDA errors)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--force_cpu', action='store_true', 
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--batch_limit', type=int, default=None,
                        help='Limit the number of batches processed per epoch (for debugging)')
    parser.add_argument('--margin', type=int, default=10,
                        help='Number of pixels to discard from the edge of each sample to reduce overfitting (0-20)')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, args):
    """
    Train the model with the given parameters
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        args: Command line arguments
        
    Returns:
        Trained model and training history
    """
    print(f"Setting up training environment...")
    device = torch.device(args.device)
    model = model.to(device)
    
    # Initialize learning rate scheduler
    if args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler with patience=5, factor=0.5")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min=1e-6")
    else:
        scheduler = None
        print("Not using any learning rate scheduler")
    
    # Initialize mixed precision scaler if using mixed precision
    scaler = amp.GradScaler() if args.mixed_precision else None
    if scaler:
        print("Initialized mixed precision scaler")
        
    # For early stopping and model saving
    best_val_metrics = {'loss': float('inf'), 'accuracy': 0, 'f1': 0}
    best_model_states = {'loss': None, 'accuracy': None, 'f1': None}
    patience_counter = 0
    
    # History to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # Progressive unfreezing configuration
    if args.progressive_unfreezing:
        frozen_layers = list(model.features.children())
        # Initially freeze all feature layers and train only classifier
        for param in model.features.parameters():
            param.requires_grad = False
        print("Initial state: All feature layers frozen - only training classifier")
        print(f"Will unfreeze {len(frozen_layers)} layer groups progressively during training")
        layers_unfrozen = 0
    
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    
    try:
        # Check if at least one batch can be loaded from training data
        print("Testing data loader (first batch)...")
        test_batch = next(iter(train_loader))
        
        # Print shape information
        if isinstance(test_batch['volume'], torch.Tensor):
            # If volume is already a tensor (batched by custom_collate_fn)
            # print(f"First batch loaded successfully! Volume shape: {test_batch['volume'].shape}")
            volumes = test_batch['volume'].to(device)
        else:
            # If volume is still a list
            # print(f"First batch loaded successfully! First volume shape: {test_batch['volume'][0].shape}")
            volumes = torch.stack(test_batch['volume']).to(device)
        
        # Convert labels to tensor and move to device
        labels = torch.tensor(test_batch['label']).to(device)
        print(f"Labels shape: {labels.shape}")
        print(f"Label values: {labels.cpu().tolist()}")
        
        # Verify labels are within expected range
        n_classes = model.classifier[-1].out_features
        min_class = labels.min().item()
        max_class = labels.max().item()
        print(f"Label range: [{min_class}, {max_class}], Number of model output classes: {n_classes}")
        
        if min_class < 0 or max_class >= n_classes:
            print(f"WARNING: Labels out of range! Must be between 0 and {n_classes-1}")
            if args.force_cpu:
                # Continue anyway since we're in debug mode
                print("Continuing anyway as we're in debug mode")
            else:
                # Only continue if in debug mode or specified by user
                if '--force_continue' not in sys.argv:
                    print("Aborting due to label mismatch. Use --force_continue to override.")
                    return None, None
        
        # Try a sample forward pass to validate the model
        print("Performing sample forward pass...")
        with torch.no_grad():
            outputs = model(volumes)
            print(f"Model output shape: {outputs.shape}")
        
        # Free memory
        del volumes, labels
        torch.cuda.empty_cache()
        print("Initial validation successful!")
    except Exception as e:
        print(f"ERROR loading first batch: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    # Flag to track if we need to adjust labels from 1-based to 0-based indexing
    adjust_labels = False
    if min_class == 1:
        adjust_labels = True
        print("Will adjust labels by subtracting 1 to convert from 1-based to 0-based indexing")
        
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        
        # Unfreeze layers progressively
        if args.progressive_unfreezing and epoch > 0 and epoch % 3 == 0 and layers_unfrozen < len(frozen_layers):
            # Unfreeze the last frozen layer group
            layer_to_unfreeze = frozen_layers[-(1 + layers_unfrozen)]
            for param in layer_to_unfreeze.parameters():
                param.requires_grad = True
            layers_unfrozen += 1
            print(f"Epoch {epoch+1}/{args.epochs}: Unfrozen another layer group - {layers_unfrozen}/{len(frozen_layers)}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Training phase started...")
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        batch_count = 0
        
        for batch in progress_bar:
            try:
                batch_count += 1
                # Get data
                volumes = batch['volume']
                labels = batch['label']
                
                # Validate data before moving to device
                if isinstance(volumes, list) and (len(volumes) == 0 or len(labels) == 0):
                    print(f"WARNING: Empty batch {batch_count} - skipping")
                    continue
                
                # Check if volumes is already stacked (5D tensor) or still a list
                if isinstance(volumes, torch.Tensor):
                    # Already stacked by custom_collate_fn
                    if volumes.dim() != 5:
                        print(f"WARNING: Expected 5D tensor but got shape {volumes.shape} for batch {batch_count}")
                        # Try to fix the shape if needed
                        if volumes.dim() == 4:  # (B, D, H, W) - missing channel dimension
                            volumes = volumes.unsqueeze(1)  # Add channel -> (B, 1, D, H, W)
                    
                    # Check for NaN/Inf values before sending to GPU
                    if torch.isnan(volumes).any() or torch.isinf(volumes).any():
                        print(f"WARNING: NaN/Inf values in volumes for batch {batch_count}")
                        volumes = torch.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Move to device
                    volumes = volumes.to(device)
                    if adjust_labels and isinstance(labels, list):
                        # Convert labels from 1-based to 0-based indexing
                        labels = [l - 1 for l in labels]
                    labels = torch.tensor(labels).to(device)
                else:
                    # Check for uniform shapes in list
                    if len(volumes) > 0:
                        first_shape = volumes[0].shape
                        if not all(v.shape == first_shape for v in volumes):
                            print(f"WARNING: Non-uniform volume shapes in batch {batch_count}")
                            print(f"Shapes: {[v.shape for v in volumes]}")
                            print("Skipping this batch")
                            continue
                    
                    # Move to device with additional error handling
                    try:
                        volumes_stacked = torch.stack(volumes)
                        # Check for NaN/Inf values before sending to GPU
                        if torch.isnan(volumes_stacked).any() or torch.isinf(volumes_stacked).any():
                            print(f"WARNING: NaN/Inf values in volumes_stacked for batch {batch_count}")
                            volumes_stacked = torch.nan_to_num(volumes_stacked, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        volumes = volumes_stacked.to(device)
                        if adjust_labels and isinstance(labels, list):
                            # Convert labels from 1-based to 0-based indexing
                            labels = [l - 1 for l in labels]
                        labels = torch.tensor(labels).to(device)
                    except RuntimeError as e:
                        print(f"ERROR stacking volumes in batch {batch_count}: {e}")
                        print(f"Volume shapes: {[v.shape for v in volumes]}")
                        print(f"Skipping this batch and continuing...")
                        continue
                
                # Forward pass with mixed precision if enabled
                optimizer.zero_grad()
                
                try:
                    if args.mixed_precision:
                        with amp.autocast():
                            outputs = model(volumes)
                            loss = criterion(outputs, labels)
                        
                        # Backward pass and optimize with mixed precision
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Regular forward/backward pass
                        outputs = model(volumes)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    # Track statistics
                    batch_loss = loss.item()
                    train_loss += batch_loss * volumes.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({'loss': batch_loss, 'acc': train_correct/train_total if train_total > 0 else 0})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: CUDA out of memory in batch {batch_count}")
                        # Print memory stats
                        if torch.cuda.is_available():
                            print(f"  CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                            print(f"  CUDA memory cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
                        
                        # Clear caches and try to continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Skip this batch
                        continue
                    else:
                        # Re-raise other runtime errors
                        print(f"ERROR in forward/backward pass for batch {batch_count}: {e}")
                        if "device-side assert" in str(e):
                            print("This could be due to invalid inputs or labels. Consider using CUDA_LAUNCH_BLOCKING=1")
                            # Try to print the problematic data
                            if isinstance(volumes, torch.Tensor):
                                print(f"  Volumes shape: {volumes.shape}")
                                print(f"  Labels: {labels.tolist()}")
                        
                        # Skip this batch instead of raising
                        continue
                
                # Free up memory
                del volumes, labels, outputs
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"ERROR in training batch {batch_count}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        train_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        
        # Validation phase - run every validation_freq epochs or on last epoch
        if (epoch % args.validation_freq == 0) or (epoch == args.epochs - 1):
            print(f"Validation phase started...")
            try:
                val_metrics = validate_model(model, val_loader, criterion, device, adjust_labels)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                val_f1 = val_metrics['f1']
                
                # Print validation results
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Update history
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                
                # Check for improvements and save models based on different metrics
                improved = False
                for metric in args.save_best_metrics:
                    if metric == 'loss' and val_loss < best_val_metrics['loss']:
                        best_val_metrics['loss'] = val_loss
                        best_model_states['loss'] = model.state_dict().copy()
                        improved = True
                        
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_best_loss.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_states['loss'],
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'val_f1': val_f1
                        }, checkpoint_path)
                        print(f"  Saved best loss model to {checkpoint_path}")
                    
                    if metric == 'accuracy' and val_acc > best_val_metrics['accuracy']:
                        best_val_metrics['accuracy'] = val_acc
                        best_model_states['accuracy'] = model.state_dict().copy()
                        improved = True
                        
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_best_accuracy.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_states['accuracy'],
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'val_f1': val_f1
                        }, checkpoint_path)
                        print(f"  Saved best accuracy model to {checkpoint_path}")
                    
                    if metric == 'f1' and val_f1 > best_val_metrics['f1']:
                        best_val_metrics['f1'] = val_f1
                        best_model_states['f1'] = model.state_dict().copy()
                        improved = True
                        
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_best_f1.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_states['f1'],
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'val_f1': val_f1
                        }, checkpoint_path)
                        print(f"  Saved best F1 model to {checkpoint_path}")
                
                # Update early stopping counter
                if improved:
                    patience_counter = 0
                    print(f"  Found improvement in tracked metrics!")
                else:
                    patience_counter += 1
                    print(f"  No improvement in tracked metrics. Patience: {patience_counter}/{args.early_stopping}")
            except Exception as e:
                print(f"ERROR during validation: {e}")
                import traceback
                traceback.print_exc()
                # Continue training even if validation fails
                history['val_loss'].append(None)
                history['val_acc'].append(None)
                history['val_f1'].append(None)
        else:
            # Even if we don't validate, add placeholders to keep history aligned
            history['val_loss'].append(None)
            history['val_acc'].append(None)
            history['val_f1'].append(None)
            print(f"  Skipping validation this epoch (will validate every {args.validation_freq} epochs)")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Update learning rate with scheduler if enabled
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        print(f"  Current learning rate: {current_lr:.6f}")
        
        if scheduler is not None:
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss if (epoch % args.validation_freq == 0) else train_loss)
            else:
                scheduler.step()
            
            if optimizer.param_groups[0]['lr'] < current_lr:
                print(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
        
        # Clear memory at the end of each epoch
        torch.cuda.empty_cache()
    
    # Choose the best model based on validation loss if available
    best_metric = args.save_best_metrics[0]  # Default to first specified metric
    if best_model_states[best_metric] is not None:
        model.load_state_dict(best_model_states[best_metric])
        print(f"Loaded best model based on {best_metric}: {best_val_metrics[best_metric]:.4f}")
    
    return model, history

def validate_model(model, data_loader, criterion, device, adjust_labels=False):
    """
    Validate the model on the provided data loader
    
    Args:
        model: The model to validate
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
        adjust_labels: Whether to adjust labels from 1-based to 0-based indexing
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    # Check if we should force CPU validation based on the global args context
    force_cpu_validation = False
    if 'args' in globals() and hasattr(globals()['args'], 'force_cpu_validation'):
        force_cpu_validation = globals()['args'].force_cpu_validation
    
    # Force CPU validation if requested
    if force_cpu_validation:
        print("Forcing validation on CPU as requested")
        validation_device = torch.device('cpu')
        # Move model to CPU first
        model = model.cpu()
    else:
        validation_device = device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Validating')):
            try:
                # Get data
                volumes = batch['volume']
                labels = batch['label']
                
                # Validate data before processing
                if isinstance(volumes, list) and (len(volumes) == 0 or len(labels) == 0):
                    print(f"WARNING: Empty validation batch {batch_idx} - skipping")
                    continue
                
                # Check if volumes is already stacked (5D tensor) or still a list
                if isinstance(volumes, torch.Tensor):
                    # Already stacked by custom_collate_fn
                    if volumes.dim() != 5:
                        print(f"WARNING: Expected 5D tensor but got shape {volumes.shape} for batch {batch_idx}")
                        # Try to fix the shape if needed
                        if volumes.dim() == 4:  # (B, D, H, W) - missing channel dimension
                            volumes = volumes.unsqueeze(1)  # Add channel -> (B, 1, D, H, W)
                    
                    # Check for NaN/Inf values
                    if torch.isnan(volumes).any() or torch.isinf(volumes).any():
                        print(f"WARNING: NaN/Inf values in validation batch {batch_idx}")
                        volumes = torch.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    try:    
                        # Move to validation device - either CPU or original device
                        volumes = volumes.to(validation_device)
                        if adjust_labels and isinstance(labels, list):
                            # Convert labels from 1-based to 0-based indexing
                            labels = [l - 1 for l in labels]
                        labels = torch.tensor(labels).to(validation_device)
                    except RuntimeError as e:
                        print(f"ERROR moving batch {batch_idx} to device: {e}")
                        print("Falling back to CPU for this batch")
                        volumes = volumes.cpu()
                        if adjust_labels and isinstance(labels, list):
                            # Convert labels from 1-based to 0-based indexing
                            labels = [l - 1 for l in labels]
                        labels = torch.tensor(labels).cpu()
                else:
                    # Check for uniform shapes in list
                    if len(volumes) > 0:
                        first_shape = volumes[0].shape
                        if not all(v.shape == first_shape for v in volumes):
                            print(f"WARNING: Non-uniform volume shapes in validation batch {batch_idx}")
                            print(f"Shapes: {[v.shape for v in volumes]}")
                            print("Skipping this batch")
                            continue
                    
                    # Move to validation device with error handling
                    try:
                        volumes_stacked = torch.stack(volumes)
                        # Check for NaN/Inf values
                        if torch.isnan(volumes_stacked).any() or torch.isinf(volumes_stacked).any():
                            print(f"WARNING: NaN/Inf values in validation batch {batch_idx}")
                            volumes_stacked = torch.nan_to_num(volumes_stacked, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        volumes = volumes_stacked.to(validation_device)
                        if adjust_labels and isinstance(labels, list):
                            # Convert labels from 1-based to 0-based indexing
                            labels = [l - 1 for l in labels]
                        labels = torch.tensor(labels).to(validation_device)
                    except RuntimeError as e:
                        print(f"ERROR stacking volumes in validation batch {batch_idx}: {e}")
                        print(f"Volume shapes: {[v.shape for v in volumes]}")
                        print(f"Skipping this batch and continuing...")
                        continue
                
                # Verify labels are within expected range
                n_classes = model.classifier[-1].out_features
                if labels.min() < 0 or labels.max() >= n_classes:
                    print(f"WARNING: Labels out of range in batch {batch_idx}!")
                    print(f"Label values: {labels.tolist()}")
                    print(f"Must be between 0 and {n_classes-1}")
                    print("Skipping this batch")
                    continue
                
                try:
                    # Forward pass
                    outputs = model(volumes)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * volumes.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Save predictions and labels for metrics
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except RuntimeError as e:
                    print(f"ERROR in forward pass for validation batch {batch_idx}: {e}")
                    if "device-side assert" in str(e):
                        print("This could be due to invalid inputs or labels.")
                        print(f"  Volumes shape: {volumes.shape}")
                        print(f"  Labels: {labels.tolist()}")
                    
                    # Skip this batch and continue
                    continue
                
                # Free up memory
                del volumes, labels, outputs
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"ERROR in validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Handle case where no batches could be processed
    if val_total == 0:
        print("WARNING: No validation batches could be processed!")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'f1': 0.0,
            'predictions': [],
            'labels': []
        }
    
    # Calculate metrics
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    # Calculate F1 score (macro average)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    val_f1 = report['macro avg']['f1-score']
    
    # Move model back to original device if needed
    if force_cpu_validation and device.type == 'cuda':
        model = model.to(device)
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'f1': val_f1,
        'predictions': all_preds,
        'labels': all_labels
    }

def plot_training_history(history, save_path):
    """
    Plot the training history with enhanced visualizations
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot loss
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history['train_loss'], label='Training Loss', color='blue', linewidth=2)
    # Filter out None values for validation metrics
    val_epochs = [i for i, x in enumerate(history['val_loss']) if x is not None]
    val_loss = [x for x in history['val_loss'] if x is not None]
    ax1.plot(val_epochs, val_loss, label='Validation Loss', color='red', linewidth=2, marker='o', markersize=5)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue', linewidth=2)
    val_acc = [x for x in history['val_acc'] if x is not None]
    ax2.plot(val_epochs, val_acc, label='Validation Accuracy', color='red', linewidth=2, marker='o', markersize=5)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot F1 score
    ax3 = fig.add_subplot(2, 2, 3)
    val_f1 = [x for x in history['val_f1'] if x is not None]
    ax3.plot(val_epochs, val_f1, label='Validation F1 Score', color='purple', linewidth=2, marker='o', markersize=5)
    ax3.set_title('F1 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(history['learning_rates'], label='Learning Rate', color='green', linewidth=2)
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    if len(history['learning_rates']) > 1:
        ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved enhanced training history plot to {save_path}')

def evaluate_model(model, test_loader, args, adjust_labels=False):
    """
    Evaluate the model on the test set with enhanced metrics
    
    Args:
        model: The trained model
        test_loader: DataLoader for test data
        args: Command line arguments
        adjust_labels: Whether to adjust labels from 1-based to 0-based indexing
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_sample_ids = []
    all_probs = []  # Store prediction probabilities
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # Get data
            volumes = batch['volume']
            labels = batch['label']
            sample_ids = batch['metadata']['sample_id']
            
            # Move to device
            volumes = torch.stack(volumes).to(device)
            if adjust_labels and isinstance(labels, list):
                # Convert labels from 1-based to 0-based indexing
                labels = [l - 1 for l in labels]
            labels = torch.tensor(labels).to(device)
            
            # Forward pass
            outputs = model(volumes)
            _, predicted = torch.max(outputs, 1)
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sample_ids.extend(sample_ids)
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Calculate top-k accuracy (k=2) for multi-class problems
    if len(np.unique(all_labels)) > 2:
        top2_correct = 0
        for probs, true_label in zip(all_probs, all_labels):
            top2_indices = np.argsort(probs)[-2:]  # Get indices of top 2 probabilities
            if true_label in top2_indices:
                top2_correct += 1
        top2_accuracy = top2_correct / len(all_labels)
    else:
        top2_accuracy = None
    
    # Create results dictionary with enhanced metrics
    results = {
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': list(zip(all_sample_ids, all_labels, all_preds)),
        'top2_accuracy': top2_accuracy,
        'prediction_probs': all_probs
    }
    
    # Print summary
    print(f"Evaluation Results:")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    if top2_accuracy:
        print(f"  Top-2 Accuracy: {top2_accuracy:.4f}")
    
    return results

def save_evaluation_results(results, save_dir, class_names=None):
    """
    Save evaluation results to files with improved visualizations
    
    Args:
        results: Dictionary with evaluation metrics
        save_dir: Directory to save results
        class_names: Optional list of class names for better labels
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique classes
    classes = np.unique(np.array([label for _, label, _ in results['predictions']]))
    
    # Use provided class names or default to class indices
    if class_names is None or len(class_names) != len(classes):
        class_names = [f'Class {c}' for c in classes]
    
    # Save confusion matrix plot with improved visualization
    plt.figure(figsize=(12, 10))
    cm = results['confusion_matrix']
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the normalized confusion matrix
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                annot_kws={"size": 10}, cbar=True)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    
    # Also plot raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                annot_kws={"size": 10}, cbar=True)
    plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    
    # Save per-class precision, recall and F1 scores as a bar chart
    report = results['classification_report']
    plt.figure(figsize=(15, 8))
    
    metrics = ['precision', 'recall', 'f1-score']
    values = {m: [] for m in metrics}
    for c in classes:
        for m in metrics:
            values[m].append(report[str(c)][m])
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width, values['precision'], width, label='Precision', color='skyblue')
    rects2 = ax.bar(x, values['recall'], width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + width, values['f1-score'], width, label='F1-score', color='salmon')
    
    ax.set_xlabel('Class', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Per-class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line for average values
    avg_precision = report['macro avg']['precision']
    avg_recall = report['macro avg']['recall']
    avg_f1 = report['macro avg']['f1-score']
    
    ax.axhline(y=avg_precision, color='blue', linestyle='--', alpha=0.5, label='Avg Precision')
    ax.axhline(y=avg_recall, color='green', linestyle='--', alpha=0.5, label='Avg Recall')
    ax.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.5, label='Avg F1')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    
    # Save predictions to CSV
    with open(os.path.join(save_dir, 'predictions.csv'), 'w') as f:
        f.write('sample_id,true_label,predicted_label\n')
        for sample_id, true_label, pred_label in results['predictions']:
            f.write(f'{sample_id},{true_label},{pred_label}\n')
    
    # Save detailed classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write('Class\tPrecision\tRecall\tF1-Score\tSupport\n')
        for class_id, class_name in zip(classes, class_names):
            metrics = report[str(class_id)]
            f.write(f"{class_name}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1-score']:.4f}\t{metrics['support']}\n")
        
        f.write(f"\nAccuracy: {report['accuracy']:.4f}\n")
        f.write(f"Macro Avg: {report['macro avg']['precision']:.4f}\t{report['macro avg']['recall']:.4f}\t{report['macro avg']['f1-score']:.4f}\t{report['macro avg']['support']}\n")
        f.write(f"Weighted Avg: {report['weighted avg']['precision']:.4f}\t{report['weighted avg']['recall']:.4f}\t{report['weighted avg']['f1-score']:.4f}\t{report['weighted avg']['support']}\n")
    
    print(f'Saved enhanced evaluation results to {save_dir}')

def basic_transform(x, margin=0):
    """Transform a 3D volume to a 5D tensor with shape (1, 1, D, H, W)"""
    # First convert to numpy array if not already
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Apply margin if specified (discard pixels from edges)
    if margin > 0 and x.ndim == 3:
        # Ensure margin isn't too large for the input
        safe_margin = min(margin, min(x.shape) // 4)
        if safe_margin > 0:
            # Apply margin to all three dimensions
            x = x[safe_margin:-safe_margin, safe_margin:-safe_margin, safe_margin:-safe_margin]
    
    # Convert to tensor
    x_tensor = torch.from_numpy(x).float()
    
    # Normalize to [0, 1] if needed
    if x_tensor.max() > 1.0:
        x_tensor = x_tensor / 255.0
    
    # Get the original shape
    orig_shape = x_tensor.shape
    
    # Reshape to the expected 5D format (1, 1, D, H, W)
    if len(orig_shape) == 3:  # (D, H, W)
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
    elif len(orig_shape) == 4:  # (C, D, H, W) or (B, D, H, W)
        # Assume first dimension is channels
        x_tensor = x_tensor.unsqueeze(0)  # -> (1, C, D, H, W)
        if x_tensor.shape[1] != 1:
            # If more than 1 channel, use only the first one
            x_tensor = x_tensor[:, 0:1, ...]  # Keep only the first channel
    elif len(orig_shape) == 5:  # Already (B, C, D, H, W)
        if x_tensor.shape[1] != 1:
            # If more than 1 channel, use only the first one
            x_tensor = x_tensor[:, 0:1, ...]
    else:
        # Handle unexpected shape
        print(f"WARNING: Unexpected tensor shape in basic_transform: {orig_shape}")
        # Reshape to expected format by flattening and reshaping
        try:
            # Extract D, H, W from crop_size
            D, H, W = 80, 80, 80  # Default size
            x_tensor = x_tensor.reshape(1, 1, D, H, W)
        except:
            # If reshaping fails, create a new zero tensor
            x_tensor = torch.zeros((1, 1, 80, 80, 80), dtype=torch.float32)
    
    # Final verification
    if x_tensor.shape[0] != 1 or x_tensor.shape[1] != 1 or x_tensor.dim() != 5:
        print(f"ERROR: Failed to properly shape tensor. Current shape: {x_tensor.shape}")
        # Last resort - force reshape to expected dimensions
        x_tensor = torch.zeros((1, 1, 80, 80, 80), dtype=torch.float32)
    
    return x_tensor

def augmented_transform(x, margin=0):
    """Transform a 3D volume to a 5D tensor with shape (1, 1, D, H, W) with augmentations"""
    # First apply basic transform to get initial 5D tensor
    x_tensor = basic_transform(x, margin)
    
    # Apply augmentations with 50% chance for each
    if np.random.rand() < 0.5:  # Random flip along z-axis
        x_tensor = torch.flip(x_tensor, dims=[2])
        
    if np.random.rand() < 0.5:  # Random flip along y-axis
        x_tensor = torch.flip(x_tensor, dims=[3])
        
    if np.random.rand() < 0.5:  # Random flip along x-axis
        x_tensor = torch.flip(x_tensor, dims=[4])
        
    # Random rotation (90, 180, 270 degrees) in xy plane
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)  # 1, 2, or 3 for 90, 180, 270 degrees
        x_tensor = torch.rot90(x_tensor, k, dims=[3, 4])
        
    # Random intensity adjustments
    if np.random.rand() < 0.5:
        # Random gamma correction
        gamma = np.random.uniform(0.8, 1.2)
        x_tensor = x_tensor ** gamma
        
    # Random contrast adjustment
    if np.random.rand() < 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        mean_val = x_tensor.mean()
        x_tensor = (x_tensor - mean_val) * contrast + mean_val
        x_tensor = torch.clamp(x_tensor, 0, 1)
        
    # Random noise addition
    if np.random.rand() < 0.3:
        noise_level = np.random.uniform(0, 0.05)
        noise = torch.randn_like(x_tensor) * noise_level
        x_tensor = x_tensor + noise
        x_tensor = torch.clamp(x_tensor, 0, 1)
    
    # Final check for shape
    if x_tensor.shape[0] != 1 or x_tensor.shape[1] != 1 or x_tensor.dim() != 5:
        print(f"WARNING: augmented_transform produced incorrect shape: {x_tensor.shape}")
        # Force correct shape
        x_tensor = basic_transform(x.cpu().numpy() if torch.is_tensor(x) else x)
        
    return x_tensor

def mask_transform(x):
    """Transform a mask to a 5D tensor with shape (1, 1, D, H, W)"""
    if x is None:
        return None
    
    # Use basic transform for masks as well, but without normalization
    x_tensor = torch.from_numpy(x).float()
    
    # Reshape to the expected 5D format (1, 1, D, H, W)
    if x_tensor.dim() == 3:  # (D, H, W)
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
    elif x_tensor.dim() == 4:  # (C, D, H, W) or (B, D, H, W)
        # Assume first dimension is channels
        x_tensor = x_tensor.unsqueeze(0)  # -> (1, C, D, H, W)
        if x_tensor.shape[1] != 1:
            # If more than 1 channel, use only the first one
            x_tensor = x_tensor[:, 0:1, ...]
    elif x_tensor.dim() != 5:
        # Handle unexpected shape
        print(f"WARNING: Unexpected mask tensor shape: {x_tensor.shape}")
        # Extract dimensions
        D, H, W = 80, 80, 80  # Default
        if len(x_tensor.shape) >= 3:
            D, H, W = x_tensor.shape[-3:]
        # Force reshape
        x_tensor = x_tensor.reshape(1, 1, D, H, W)
    
    return x_tensor

def get_transforms(target_size, use_augmentation=False, margin=0):
    """
    Define transformations for the input data.
    
    Args:
        target_size: Target size for volumes (depth, height, width)
        use_augmentation: Whether to use data augmentation
        margin: Number of pixels to discard from each edge
        
    Returns:
        transform, mask_transform functions
    """
    if use_augmentation:
        return partial(augmented_transform, margin=margin), mask_transform
    else:
        return partial(basic_transform, margin=margin), mask_transform

def monitor_memory(stop_event, interval=5.0):
    """
    Monitor system and GPU memory usage periodically
    Args:
        stop_event: Threading event to signal when to stop monitoring
        interval: Time between memory checks in seconds
    """
    print("Starting memory monitor...")
    while not stop_event.is_set():
        # System memory
        process = psutil.Process(os.getpid())
        sys_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # GPU memory if available
        gpu_mem_allocated = 0
        gpu_mem_cached = 0
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_mem_cached = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
        print(f"\nMEMORY MONITOR:")
        print(f"  System memory used: {sys_mem:.2f} MB")
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {gpu_mem_allocated:.2f} MB")
            print(f"  GPU memory reserved: {gpu_mem_cached:.2f} MB")
            
        # Sleep for the interval
        time.sleep(interval)

def non_augmented_transform(x, target_size, margin=0):
    """Non-augmented transform function that ensures proper 5D tensor output"""
    return basic_transform(x, margin)

def ensure_index_csv_exists(args):
    """
    Ensure the nuclei index CSV file exists, creating it if needed.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Path to the index CSV file
    """
    if not os.path.exists(args.index_csv) or args.create_index:
        print(f"Creating nuclei index CSV file at {args.index_csv}...")
        # Import here to avoid circular imports
        from scripts.create_nuclei_index import create_nuclei_index
        create_nuclei_index(args.data_dir, args.index_csv, args.class_csv)
    else:
        print(f"Using existing nuclei index CSV file: {args.index_csv}")
    
    return args.index_csv

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set CUDA options
    if args.force_cpu:
        args.device = 'cpu'
        print("Forcing CPU usage regardless of CUDA availability")
    
    if args.cuda_launch_blocking:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("Set CUDA_LAUNCH_BLOCKING=1 for better error reporting")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure the index CSV file exists
    index_csv = ensure_index_csv_exists(args)
    
    # Get transforms
    transform, mask_transform = get_transforms(args.target_size, args.use_augmentation, args.margin)
    
    if args.margin > 0:
        print(f"Using margin of {args.margin} pixels to discard from edges of samples")
    
    # Make sure the nuclei index file exists
    ensure_index_csv_exists(args)
    
    print(f"Setting up data loaders...")
    
    # Start memory monitoring in a separate thread if requested
    stop_monitor = threading.Event()
    monitor_thread = None
    if args.memory_monitor:
        monitor_thread = threading.Thread(
            target=monitor_memory, 
            args=(stop_monitor, args.monitor_interval)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        print(f"Started memory monitoring with interval {args.monitor_interval}s")
    
    # Load data using optimized lazy loading
    try:
        # Get train and validation dataloaders
        print("Creating data loaders with lazy loading...")
        
        dataloader = get_nuclei_dataloader(
                root_dir=args.data_dir,
                batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
                transform=transform,
                mask_transform=mask_transform,
                class_csv_path=args.class_csv,
                filter_by_class=args.class_id,
            ignore_unclassified=True,
                target_size=tuple(args.target_size),
                sample_percent=args.sample_percent,
            pin_memory=args.pin_memory,
            debug=True  # Enable debug output for the dataloader
        )
        
        # Split into train and validation
        dataset_size = len(dataloader.dataset)
        train_size = int(dataset_size * args.train_split)
        val_size = dataset_size - train_size
        
        # Create random split indices
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create new dataloaders with the split indices
        from torch.utils.data import Subset
        train_dataset = Subset(dataloader.dataset, train_indices)
        val_dataset = Subset(dataloader.dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
                batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=args.pin_memory,
            persistent_workers=False if args.num_workers == 0 else True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=args.pin_memory,
            persistent_workers=False if args.num_workers == 0 else True
        )
        
        print(f"Created data loaders: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
        
        # Determine the number of output classes
        if args.output_classes is None:
            # Get class information from the dataset
            all_classes = set()
            
            # Load from the class CSV if available
            if args.class_csv and os.path.exists(args.class_csv):
                class_df = pd.read_csv(args.class_csv)
                if args.class_id is not None:
                    if isinstance(args.class_id, int):
                        args.class_id = [args.class_id]
                    class_df = class_df[class_df['class_id'].isin(args.class_id)]
                all_classes = set(class_df['class_id'].unique())
            
            if not all_classes:  # Fallback if class CSV is not available
                # Get a batch from the dataloader and check the labels
                for batch in train_loader:
                    if isinstance(batch['label'], list):
                        all_classes.update(batch['label'])
                    else:
                        all_classes.update(batch['label'].cpu().numpy())
                    break
            
            # Handle class IDs that start from 1 instead of 0
            min_class = min(all_classes) if all_classes else 1
            max_class = max(all_classes) if all_classes else 19
            
            if min_class == 1:
                # Class IDs start from 1, so we need exactly max_class output classes
                args.output_classes = max_class
                print(f"Classes start from 1. Setting output classes to match max class ID: {args.output_classes}")
                
                # Add a note about how labels will be handled
                print("NOTE: Class IDs in CSV file start from 1, but model inputs will be adjusted to start from 0")
            else:
                # Standard case: add 1 because class IDs start from 0
                args.output_classes = max_class + 1
                print(f"Automatically determined number of output classes: {args.output_classes}")
        
        # Load model
        print(f"Loading model from checkpoint: {args.checkpoint}")
        
        # Create a base model first to extract features
        base_model = Vgg3D(
            input_size=tuple(args.target_size),
            output_classes=args.output_classes,
            input_fmaps=1  # Single channel input
        )
        
        # Load only the feature extractor weights
        try:
            print("Loading checkpoint (features only)...")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            # Extract the state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Filter to include only feature layers
            feature_state_dict = {k: v for k, v in state_dict.items() if k.startswith('features')}
            base_model.load_state_dict(feature_state_dict, strict=False)
            print(f"Successfully loaded feature extractor weights from checkpoint")
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            print("Proceeding with randomly initialized feature extractor")
            
        # Get an actual sample from the dataset to determine correct dimensions 
        print("Getting a sample from the dataset to determine actual dimensions...")
        try:
            # Get a sample from the training dataloader
            sample_batch = next(iter(train_loader))
            sample_volume = sample_batch['volume']
            
            # Convert to tensor if it's a list
            if isinstance(sample_volume, list):
                sample_volume = torch.stack(sample_volume)
                
            # Extract a single sample if it's a batch
            if len(sample_volume.shape) > 4:
                sample_volume = sample_volume[0:1]
                
            print(f"Sample volume shape from dataset: {sample_volume.shape}")
            
            # Run it through the feature extractor to get the actual output size
            with torch.no_grad():
                features = base_model.features(sample_volume)
                feature_size = features.view(features.size(0), -1).size(1)
            
            print(f"Actual feature output size with real sample: {feature_size}")
        except Exception as e:
            print(f"Error getting real sample dimensions: {e}")
            # Fall back to creating a synthetic sample
            print("Falling back to synthetic sample...")
            # Apply the exact same transformations that would happen in the dataloader
            synthetic_sample = torch.zeros(1, 1, *args.target_size)
            if args.margin > 0:
                margin = args.margin
                print(f"Applying margin of {margin} to synthetic sample")
                synthetic_sample = synthetic_sample[:, :, margin:-margin, margin:-margin, margin:-margin]
            print(f"Synthetic sample shape: {synthetic_sample.shape}")
            
            with torch.no_grad():
                features = base_model.features(synthetic_sample)
                feature_size = features.view(features.size(0), -1).size(1)
            
            print(f"Feature size with synthetic sample: {feature_size}")
        
        # Create the adapter model with the correct feature output size -> classifier input size
        print("Creating adapter model with appropriate classifier size...")
        
        # Create a custom Vgg3DAdapter with the correct feature size
        class CustomVgg3DAdapter(nn.Module):
            def __init__(self, base_features, feature_size, num_classes, classifier_size=512):
                super(CustomVgg3DAdapter, self).__init__()
                self.features = base_features
                
                self.classifier = nn.Sequential(
                    nn.Linear(feature_size, classifier_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(classifier_size, classifier_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(classifier_size, num_classes),
                )
                
                # Initialize the classifier
                for m in self.classifier.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        # Create the adapter with the exact feature size needed
        model = CustomVgg3DAdapter(
            base_features=base_model.features,
            feature_size=feature_size,
            num_classes=args.output_classes,
            classifier_size=args.classifier_size
        )
        print(f"Successfully created adapter model with correct dimensions: {feature_size} -> {args.classifier_size} -> {args.output_classes}")
        
        if args.freeze_features:
            # Freeze feature extractor layers
            print("FINETUNING MODE: Freezing feature extractor layers")
            for param in model.features.parameters():
                param.requires_grad = False
            print("Only training classifier layers (significantly reduces trainable parameters)")
        else:
            print("FULL TRAINING MODE: Training all model parameters (feature extractor and classifier)")
            print("Warning: This requires significantly more training data and time compared to finetuning")
        
        if args.freeze_classifier_layers > 0:
            # Freeze the specified number of classifier layers
            print(f"Freezing {args.freeze_classifier_layers} classifier layers")
            for i, layer in enumerate(model.classifier.children()):
                if i < args.freeze_classifier_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Set up optimizer
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(
                [p for p in model.parameters() if p.requires_grad], 
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        else:  # adam
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad], 
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Check if model has any parameters to train
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        feature_params = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        
        print(f"Model has {trainable_params:,} trainable parameters")
        print(f"  - Feature extractor: {feature_params:,} trainable parameters")
        print(f"  - Classifier: {classifier_params:,} trainable parameters")
        
        # Recommend classifier reduction if high parameter count
        if classifier_params > 10000000 and not args.reduce_classifier:
            print("\nRECOMMENDATION: The classifier has a very high number of parameters.")
            print("To reduce overfitting and improve training, try running with --reduce_classifier")
            print("This will significantly reduce the number of trainable parameters.")
        
        if trainable_params == 0:
            raise ValueError("No trainable parameters in the model! Check your freeze settings.")
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train the model
        print("\nStarting training...")
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, args)
        
        # Plot and save training history
        print("\nPlotting training history...")
        plot_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
        
        # Evaluate the model on validation data
        print("\nEvaluating model on validation data...")
        val_results = evaluate_model(model, val_loader, args, adjust_labels=False)
        save_evaluation_results(val_results, args.output_dir)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
        
    except Exception as e:
        print(f"Error in training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if monitor_thread and monitor_thread.is_alive():
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)
            print("Stopped memory monitoring")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 