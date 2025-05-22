# Unfreeze all convolutional blocks (full fine-tuning)
# python scripts/finetune_vgg3d_masked.py --unfreeze_last_n_layers -1

# Unfreeze last 2 convolutional blocks (last 2 layers)
# python scripts/finetune_vgg3d_masked.py --unfreeze_last_n_layers 2


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
from dataloader.nuclei_dataloader_masked import get_masked_nuclei_dataloader
from model.vgg3d import Vgg3D, load_model_from_checkpoint

# Mask-guided Attention Module
class MaskAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskAttentionModule, self).__init__()
        self.conv = nn.Conv3d(1, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        """
        Apply mask-guided attention to features
        
        Args:
            x: Feature tensor (B, C, D, H, W)
            mask: Binary mask tensor (B, 1, D, H, W) - already resized to match x
            
        Returns:
            Attention-weighted features
        """
        # Generate attention weights from mask
        attention = self.conv(mask)
        attention = self.sigmoid(attention)
        
        # Apply attention weights to features
        return x * attention

# Custom adapter model with mask attention
class MaskedVgg3DAdapter(nn.Module):
    def __init__(self, base_model, input_size, num_classes, classifier_size=512):
        super(MaskedVgg3DAdapter, self).__init__()
        self.features = base_model.features
        self.num_classes = num_classes
        self.classifier_size = classifier_size
        self.initialized = False
        
        # Create an attention module for each feature layer
        self.attention_modules = nn.ModuleList()
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv3d):
                self.attention_modules.append(MaskAttentionModule(layer.out_channels))
        
        # We'll create the classifier during the first forward pass
        # to ensure it matches the actual dimensions of the input data
        self.classifier = None
        print("Classifier will be initialized on first forward pass to match real input dimensions")
    
    def initialize_classifier(self, feature_size):
        """Initialize the classifier based on actual feature size"""
        print(f"Initializing classifier with feature size: {feature_size}")
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, self.classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.classifier_size, self.classifier_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.classifier_size, self.num_classes),
        )
        
        # Initialize the classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Move the classifier to the same device as the features
        if next(self.features.parameters()).is_cuda:
            self.classifier = self.classifier.cuda()
        
        self.initialized = True
        return self.classifier
    
    def forward(self, x, mask=None):
        # Process features with attention if mask is provided
        if mask is not None:
            attention_idx = 0
            cur_mask = mask  # Start with original mask
            for i, layer in enumerate(self.features):
                x = layer(x)
                # Apply attention after convolutional layers
                if isinstance(layer, nn.Conv3d) and attention_idx < len(self.attention_modules):
                    # Make sure mask is resized to match current feature dimensions
                    if cur_mask.size()[2:] != x.size()[2:]:
                        cur_mask = nn.functional.interpolate(
                            cur_mask, 
                            size=x.size()[2:], 
                            mode='trilinear', 
                            align_corners=False
                        )
                    x = self.attention_modules[attention_idx](x, cur_mask)
                    attention_idx += 1
            features = x
        else:
            # Standard forward pass without mask attention
            features = self.features(x)
        
        # Flatten
        flatten_features = features.view(features.size(0), -1)
        
        # Initialize classifier on first forward pass with actual feature dimensions
        if not self.initialized:
            feature_size = flatten_features.size(1)
            self.initialize_classifier(feature_size)
        
        # Run through classifier
        return self.classifier(flatten_features)

# Mask-guided loss function
def mask_weighted_loss(outputs, labels, masks, loss_fn=nn.CrossEntropyLoss(reduction='none')):
    """
    Apply mask weighting to loss function
    
    Args:
        outputs: Model outputs (B, C) where C is number of classes
        labels: Ground truth labels (B,)
        masks: Binary masks (B, 1, D, H, W)
        loss_fn: Base loss function with reduction='none'
        
    Returns:
        Weighted loss value
    """
    # Calculate base loss per sample
    base_loss = loss_fn(outputs, labels)
    
    # Weight loss by mask values (focus on regions inside the mask)
    # Calculate mask weight as average mask value for each sample
    mask_weights = torch.mean(masks.view(masks.size(0), -1), dim=1)
    
    # Apply weights and get mean loss
    weighted_loss = base_loss * mask_weights
    return weighted_loss.mean()

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune VGG3D model on nuclei dataset with mask-guided attention and loss')
    
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
    parser.add_argument('--use_mask_attention', action='store_true', default=False,
                        help='Use mask-guided attention in the model')
    parser.add_argument('--use_mask_weighted_loss', action='store_true', default=False,
                        help='Use mask-weighted loss function')
    
    # Mask-focused dataloader options
    parser.add_argument('--use_masked_dataloader', action='store_true', default=True,
                        help='Use the mask-focused dataloader to extract subvolumes only from masked areas')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='Threshold for mask values to be considered "masked"')
    parser.add_argument('--min_masked_ratio', type=float, default=0.5,
                        help='Minimum ratio of masked voxels in a subvolume (0-1)')
    parser.add_argument('--scan_step', type=int, default=40,
                        help='Step size for scanning potential crop locations in masked dataloader')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 regularization) to reduce overfitting')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam',
                        help='Optimizer to use for training')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--freeze_features', action='store_true', default=True,
                        help='Freeze feature extractor layers and only train attention and classifier (default: True)')
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=0,
                        help='Number of later convolutional blocks to unfreeze (0 means all frozen, -1 means all unfrozen)')
    parser.add_argument('--use_gradual_unfreezing', action='store_true', default=True,
                        help='Gradually unfreeze more layers as training progresses')
    parser.add_argument('--lr_ramp_up', action='store_true', default=True,
                        help='Gradually increase learning rate with each unfreezing stage')
    parser.add_argument('--lr_ramp_factor', type=float, default=2.0,
                        help='Factor to multiply learning rate by with each unfreezing stage')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data to use for training (vs validation)')
    
    # Other params similar to original script
    parser.add_argument('--scheduler', type=str, choices=['reduce_on_plateau', 'cosine', 'none'], default='cosine',
                        help='Learning rate scheduler to use')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='Use mixed precision training (faster but may affect accuracy)')
    parser.add_argument('--validation_freq', type=int, default=1,
                        help='Validate every N epochs (default: end of each epoch)')
    parser.add_argument('--save_best_metrics', type=str, nargs='+', default=['loss', 'accuracy'],
                        choices=['loss', 'accuracy', 'f1'], help='Metrics to use for saving best models')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation during training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--margin', type=int, default=0,
                        help='Number of pixels to discard from the edge of each sample to reduce overfitting (0-20)')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, optimizer, args):
    """
    Train the model with the given parameters and mask-guided features
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        args: Command line arguments
        
    Returns:
        Trained model and training history
    """
    print(f"Setting up training environment...")
    device = torch.device(args.device)
    model = model.to(device)
    
    # Define loss function
    base_criterion = nn.CrossEntropyLoss(reduction='none' if args.use_mask_weighted_loss else 'mean')
    
    # Initialize learning rate scheduler
    if args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    
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
    
    # Prepare gradual unfreezing schedule if enabled
    if args.use_gradual_unfreezing and args.freeze_features:
        # Get all convolutional layers
        conv_layers = []
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv3d):
                conv_layers.append(i)
        
        total_blocks = len(conv_layers)
        
        # Determine unfreezing schedule
        # Start with all layers frozen, then unfreeze later layers first
        # Distribute unfreezing over epochs
        unfreeze_schedule = []
        # First epoch: no unfreezing (just classifier and attention)
        unfreeze_schedule.append(0)
        
        # Distribute remaining unfreezing across epochs 
        for epoch in range(1, args.epochs):
            blocks_to_unfreeze = int(np.ceil((epoch / (args.epochs - 1)) * total_blocks)) if args.epochs > 1 else total_blocks
            unfreeze_schedule.append(blocks_to_unfreeze)
            
        print(f"Gradual unfreezing schedule: {unfreeze_schedule}")
        
        # If also using learning rate ramping
        if args.lr_ramp_up:
            original_lr = args.lr
            lr_schedule = []
            
            for epoch in range(args.epochs):
                # Ramping factor increases with each unfreezing stage
                if epoch == 0:
                    lr_schedule.append(original_lr)
                else:
                    ramp_factor = args.lr_ramp_factor ** (epoch / (args.epochs - 1)) if args.epochs > 1 else args.lr_ramp_factor
                    lr_schedule.append(original_lr * ramp_factor)
                    
            print(f"Learning rate schedule: {[f'{lr:.6f}' for lr in lr_schedule]}")
    else:
        unfreeze_schedule = None
        lr_schedule = None
    
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Using mask-guided attention: {args.use_mask_attention}")
    print(f"Using mask-weighted loss: {args.use_mask_weighted_loss}")
    print(f"Using mask-focused dataloader: {args.use_masked_dataloader}")
    
    # Flag to track if we need to adjust labels from 1-based to 0-based indexing
    adjust_labels = False
    
    # Check for class offset in first batch
    try:
        test_batch = next(iter(train_loader))
        if test_batch is None:
            print("Warning: First batch is None, skipping test batch")
        else:
            labels = torch.tensor(test_batch['label'])
            min_class = labels.min().item()
            if min_class == 1:
                adjust_labels = True
                print("Will adjust labels by subtracting 1 to convert from 1-based to 0-based indexing")
    except Exception as e:
        print(f"Error while checking label offset: {e}")
    
    # Debug helper function to print tensor shapes
    def print_tensor_info(tensor, name):
        if tensor is None:
            print(f"{name} is None")
            return
        try:
            print(f"{name} shape: {tensor.shape}, type: {tensor.dtype}, "
                  f"min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
        except:
            print(f"{name} shape: {tensor.shape}, type: {tensor.dtype}")
    
    # Function to save a slice of a volume as an image
    def save_volume_preview(volume, mask, outputs, labels, predicted, path_prefix):
        """Save middle slices of volume and mask, and add prediction info"""
        os.makedirs(os.path.join(args.output_dir, "debug_previews"), exist_ok=True)
        
        # We'll only visualize the first sample in the batch
        # Get a middle slice for display
        if volume.dim() == 5:  # (B, C, D, H, W)
            middle_slice_idx = volume.shape[2] // 2
            volume_slice = volume[0, 0, middle_slice_idx].cpu().numpy()  # Only first sample
            mask_slice = mask[0, 0, middle_slice_idx].cpu().numpy()      # Only first sample
            
            # Get the label and prediction for the first sample only
            sample_label = labels[0].item() if labels.dim() > 0 else labels.item()
            sample_prediction = predicted[0].item() if predicted.dim() > 0 else predicted.item()
        else:
            print(f"Cannot save preview: unexpected volume shape {volume.shape}")
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Get volume statistics
        vol_min = volume_slice.min()
        vol_max = volume_slice.max()
        
        # Plot volume slice (grayscale) with EXPLICIT vmin/vmax to preserve intensity scale
        # Set vmin and vmax to 0-255 range for consistency in visualization
        display_range = (0, 255)
        im1 = ax1.imshow(volume_slice, cmap='gray', vmin=display_range[0], vmax=display_range[1])
        ax1.set_title(f"Volume (class: {sample_label}, pred: {sample_prediction})\nRAW range: {vol_min:.1f}-{vol_max:.1f}")
        plt.colorbar(im1, ax=ax1)
        
        # Plot mask slice (also grayscale since it's binary)
        im2 = ax2.imshow(mask_slice, cmap='gray', vmin=mask_slice.min(), vmax=mask_slice.max())
        ax2.set_title(f"Mask (min: {mask[0].min().item():.2f}, max: {mask[0].max().item():.2f})")
        plt.colorbar(im2, ax=ax2)
        
        # Add softmax outputs as text for the first sample only
        softmax_probs = torch.softmax(outputs, dim=1)[0].cpu().detach().numpy()  # First sample only
        text = "Softmax outputs:\n"
        for i, prob in enumerate(softmax_probs):
            if i == sample_label:
                text += f"Class {i}: {prob:.4f} (true)\n"
            elif i == sample_prediction:
                text += f"Class {i}: {prob:.4f} (pred)\n"
            elif prob > 0.01:  # Only show significant probabilities
                text += f"Class {i}: {prob:.4f}\n"
        
        fig.suptitle(text, fontsize=9, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(args.output_dir, "debug_previews", f"{path_prefix}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved input preview to {save_path} [RAW intensity range: {vol_min:.1f}-{vol_max:.1f}, Display range: {display_range}]")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        
        # Apply gradual unfreezing if enabled
        if args.use_gradual_unfreezing and args.freeze_features and unfreeze_schedule:
            blocks_to_unfreeze = unfreeze_schedule[epoch]
            
            # First freeze all feature extractor layers
            for param in model.features.parameters():
                param.requires_grad = False
            
            if blocks_to_unfreeze > 0:
                # Get all convolutional layers
                conv_layers = []
                for i, layer in enumerate(model.features):
                    if isinstance(layer, nn.Conv3d):
                        conv_layers.append(i)
                
                # Start unfreezing from the last layers (specific to this epoch)
                total_blocks = len(conv_layers)
                start_idx = max(0, total_blocks - blocks_to_unfreeze)
                
                print(f"Epoch {epoch+1}: Unfreezing {blocks_to_unfreeze} of {total_blocks} convolutional blocks")
                
                # Unfreeze the specified blocks
                for i in range(start_idx, len(conv_layers)):
                    conv_idx = conv_layers[i]
                    # Unfreeze conv layer
                    for param in model.features[conv_idx].parameters():
                        param.requires_grad = True
                    
                    # Also unfreeze batch norm
                    if conv_idx + 1 < len(model.features) and isinstance(model.features[conv_idx + 1], nn.BatchNorm3d):
                        for param in model.features[conv_idx + 1].parameters():
                            param.requires_grad = True
            else:
                print(f"Epoch {epoch+1}: All feature extractor layers frozen")
            
            # Update optimizer with new trainable parameters
            if args.optimizer == 'sgd':
                optimizer = optim.SGD(
                    [p for p in model.parameters() if p.requires_grad], 
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
            else:  # adam
                optimizer = optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], 
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
                
            # Apply learning rate ramp-up if enabled
            if args.lr_ramp_up and lr_schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[epoch]
                print(f"Learning rate updated to: {lr_schedule[epoch]:.6f}")
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Training phase started...")
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Handle case where batch could be None (returned by custom_collate_fn when all samples are invalid)
                if batch is None:
                    print(f"Warning: Batch {batch_idx} is None, skipping")
                    continue
                    
                # Get data
                volumes = batch['volume']
                masks = batch['mask']
                labels = batch['label']
                
                # # Debug info for first few batches
                # if batch_idx < 2:
                #     print(f"\nBatch {batch_idx} info:")
                #     if isinstance(volumes, list):
                #         print(f"volumes is a list of length {len(volumes)}")
                #         if len(volumes) > 0:
                #             print_tensor_info(volumes[0], "First volume")
                #     else:
                #         print_tensor_info(volumes, "volumes")
                        
                #     if isinstance(masks, list):
                #         print(f"masks is a list of length {len(masks)}")
                #         if len(masks) > 0:
                #             print_tensor_info(masks[0], "First mask")
                #     else:
                #         print_tensor_info(masks, "masks")
                
                # # Process the tensors to ensure they have correct dimensions
                if isinstance(volumes, list):
                    if len(volumes) == 0:
                        print(f"Warning: Empty volumes list in batch {batch_idx}, skipping")
                        continue
                        
                    # Check if the volumes are already 5D tensors (B,C,D,H,W)
                    if volumes[0].dim() == 5:
                        # Just stack them along the batch dimension
                        volumes = torch.cat(volumes, dim=0)
                    else:
                        # Otherwise, stack them to create a batch
                        volumes = torch.stack(volumes)
                
                if isinstance(masks, list):
                    if len(masks) == 0:
                        print(f"Warning: Empty masks list in batch {batch_idx}, skipping")
                        continue
                        
                    # Check if the masks are already 5D tensors (B,C,D,H,W)
                    if masks[0].dim() == 5:
                        # Just stack them along the batch dimension
                        masks = torch.cat(masks, dim=0)
                    else:
                        # Otherwise, stack them to create a batch
                        masks = torch.stack(masks)
                
                # # Debug the shapes after stacking
                # if batch_idx < 2:
                #     print_tensor_info(volumes, "After stacking/processing, volumes")
                #     print_tensor_info(masks, "After stacking/processing, masks")
                
                # If volume has an extra dimension (like [1,1,1,D,H,W]), remove it
                if volumes.dim() > 5:
                    volumes = volumes.squeeze(0)
                if masks.dim() > 5:
                    masks = masks.squeeze(0)
                    
                # # Debug after squeezing
                # if batch_idx < 2:
                #     print_tensor_info(volumes, "After squeezing, volumes")
                #     print_tensor_info(masks, "After squeezing, masks")
                
                # Move data to device
                volumes = volumes.to(device)
                masks = masks.to(device)
                
                # Adjust labels if needed
                if adjust_labels and isinstance(labels, list):
                    labels = [l - 1 for l in labels]
                labels = torch.tensor(labels).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Debug output after first batch
                if batch_idx == 0:
                    print_tensor_info(volumes, "Input volumes")
                    print_tensor_info(masks, "Input masks")
                    print(f"Labels: {labels}")
                
                # Forward pass with or without mask-guided attention
                try:
                    if args.use_mask_attention:
                        outputs = model(volumes, masks)
                    else:
                        outputs = model(volumes)
                    
                    if batch_idx == 0:
                        print_tensor_info(outputs, "Model outputs")
                        
                        # Save preview of the first batch
                        _, predicted = torch.max(outputs, 1)
                        if(epoch % 10 == 0):
                            save_volume_preview(volumes, masks, outputs, labels, predicted, f"epoch_{epoch+1}_batch_0_train")
                        
                except RuntimeError as e:
                    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                        # Print detailed shape information to debug
                        # print(f"Matrix multiplication error in batch {batch_idx}. Details:")
                        # print(f"  Error message: {e}")
                        # print(f"  Input volumes shape: {volumes.shape}")
                        # print(f"  Input masks shape: {masks.shape}")
                        
                        # Try running forward with debug tracing
                        print("\nRunning forward pass with step-by-step tracing:")
                        x = volumes
                        trace_mask = masks
                        
                        if args.use_mask_attention:
                            attention_idx = 0
                            for i, layer in enumerate(model.features):
                                x_shape_before = x.shape
                                x = layer(x)
                                print(f"Layer {i}: {layer.__class__.__name__} - Input shape: {x_shape_before}, Output shape: {x.shape}")
                                
                                if isinstance(layer, nn.Conv3d) and attention_idx < len(model.attention_modules):
                                    print(f"  Applying attention module {attention_idx}")
                                    print(f"  Before resize - Feature shape: {x.shape}, Mask shape: {trace_mask.shape}")
                                    
                                    if trace_mask.size()[2:] != x.size()[2:]:
                                        trace_mask = nn.functional.interpolate(
                                            trace_mask, 
                                            size=x.size()[2:], 
                                            mode='trilinear', 
                                            align_corners=False
                                        )
                                        print(f"  After resize - Mask shape: {trace_mask.shape}")
                                    
                                    x = model.attention_modules[attention_idx](x, trace_mask)
                                    print(f"  After attention - Feature shape: {x.shape}")
                                    attention_idx += 1
                            
                            print(f"Final feature shape before flatten: {x.shape}")
                            print(f"Flattened size: {x.view(x.size(0), -1).size(1)}")
                            if model.initialized:
                                print(f"Classifier expects: {model.classifier[0].in_features} features")
                            else:
                                print("Classifier not yet initialized")
                    
                    # Re-raise the error
                    raise
                
                # Calculate loss with or without mask weighting
                if args.use_mask_weighted_loss:
                    loss = mask_weighted_loss(outputs, labels, masks, base_criterion)
                else:
                    loss = base_criterion(outputs, labels)
                
                # Backward pass and optimize
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
                
                # Free up memory
                del volumes, masks, labels, outputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"ERROR in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Skip after a few consecutive failures
                if batch_idx < 5:
                    print("Continuing to next batch...")
                    continue
                else:
                    print("Too many errors, aborting training")
                    raise

        train_loss = train_loss / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        
        # Validation phase
        if (epoch % args.validation_freq == 0) or (epoch == args.epochs - 1):
            print(f"Validation phase started...")
            try:
                val_metrics = validate_model(model, val_loader, args.use_mask_attention, 
                                            args.use_mask_weighted_loss, base_criterion, device, adjust_labels, 
                                            epoch, save_preview=True, output_dir=args.output_dir)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                val_f1 = val_metrics['f1']
                
                # Print validation results
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Update history
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                
                # Check for improvements and save models
                improved = False
                for metric in args.save_best_metrics:
                    if metric == 'loss' and val_loss < best_val_metrics['loss']:
                        best_val_metrics['loss'] = val_loss
                        best_model_states['loss'] = model.state_dict().copy()
                        improved = True
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_masked_best_loss.pth')
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
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_masked_best_accuracy.pth')
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
                        
                        # Save checkpoint
                        checkpoint_path = os.path.join(args.output_dir, f'vgg3d_masked_best_f1.pth')
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
                else:
                    patience_counter += 1
                
            except Exception as e:
                print(f"ERROR during validation: {e}")
                import traceback
                traceback.print_exc()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Update learning rate with scheduler if enabled
        if scheduler is not None:
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss if epoch % args.validation_freq == 0 else train_loss)
            else:
                scheduler.step()
    
    # Choose the best model based on validation loss
    best_metric = args.save_best_metrics[0]  # Default to first specified metric
    if best_model_states[best_metric] is not None:
        model.load_state_dict(best_model_states[best_metric])
        print(f"Loaded best model based on {best_metric}: {best_val_metrics[best_metric]:.4f}")
    
    return model, history

def validate_model(model, data_loader, use_mask_attention, use_mask_weighted_loss, criterion, device, adjust_labels=False, 
                  epoch=None, save_preview=False, output_dir=None):
    """
    Validate the model on the provided data loader
    
    Args:
        model: The model to validate
        data_loader: DataLoader for validation data
        use_mask_attention: Whether to use mask-guided attention
        use_mask_weighted_loss: Whether to use mask-weighted loss
        criterion: Base loss function
        device: Device to run validation on
        adjust_labels: Whether to adjust labels from 1-based to 0-based indexing
        epoch: Current epoch number
        save_preview: Whether to save a preview of the input volume, mask, and model output
        output_dir: Directory to save previews
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    # Debug helper function
    def print_tensor_info(tensor, name):
        if tensor is None:
            print(f"{name} is None")
            return
        try:
            print(f"{name} shape: {tensor.shape}, type: {tensor.dtype}, "
                  f"min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
        except:
            print(f"{name} shape: {tensor.shape}, type: {tensor.dtype}")
    
    # Function to save a slice of a volume as an image
    def save_volume_preview(volume, mask, outputs, labels, predicted, path_prefix):
        """Save middle slices of volume and mask, and add prediction info"""
        os.makedirs(os.path.join(output_dir, "debug_previews"), exist_ok=True)
        
        # We'll only visualize the first sample in the batch
        # Get a middle slice for display
        if volume.dim() == 5:  # (B, C, D, H, W)
            middle_slice_idx = volume.shape[2] // 2
            volume_slice = volume[0, 0, middle_slice_idx].cpu().numpy()  # Only first sample
            mask_slice = mask[0, 0, middle_slice_idx].cpu().numpy()      # Only first sample
            
            # Get the label and prediction for the first sample only
            sample_label = labels[0].item() if labels.dim() > 0 else labels.item()
            sample_prediction = predicted[0].item() if predicted.dim() > 0 else predicted.item()
        else:
            print(f"Cannot save preview: unexpected volume shape {volume.shape}")
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Get volume statistics
        vol_min = volume_slice.min()
        vol_max = volume_slice.max()
        
        # Plot volume slice (grayscale) with EXPLICIT vmin/vmax to preserve intensity scale
        # Set vmin and vmax to 0-255 range for consistency in visualization
        display_range = (0, 255)
        im1 = ax1.imshow(volume_slice, cmap='gray', vmin=display_range[0], vmax=display_range[1])
        ax1.set_title(f"Volume (class: {sample_label}, pred: {sample_prediction})\nRAW range: {vol_min:.1f}-{vol_max:.1f}")
        plt.colorbar(im1, ax=ax1)
        
        # Plot mask slice (also grayscale since it's binary)
        im2 = ax2.imshow(mask_slice, cmap='gray', vmin=mask_slice.min(), vmax=mask_slice.max())
        ax2.set_title(f"Mask (min: {mask[0].min().item():.2f}, max: {mask[0].max().item():.2f})")
        plt.colorbar(im2, ax=ax2)
        
        # Add softmax outputs as text for the first sample only
        softmax_probs = torch.softmax(outputs, dim=1)[0].cpu().detach().numpy()  # First sample only
        text = "Softmax outputs:\n"
        for i, prob in enumerate(softmax_probs):
            if i == sample_label:
                text += f"Class {i}: {prob:.4f} (true)\n"
            elif i == sample_prediction:
                text += f"Class {i}: {prob:.4f} (pred)\n"
            elif prob > 0.01:  # Only show significant probabilities
                text += f"Class {i}: {prob:.4f}\n"
        
        fig.suptitle(text, fontsize=9, y=0.98)
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(output_dir, "debug_previews", f"{path_prefix}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved validation preview to {save_path} [RAW intensity range: {vol_min:.1f}-{vol_max:.1f}, Display range: {display_range}]")
    
    # Ensure model has been initialized
    if not getattr(model, 'initialized', True):
        print("Warning: Model classifier not yet initialized. Running one forward pass for initialization...")
        # Get a sample batch
        try:
            sample_batch = next(iter(data_loader))
            if sample_batch is None:
                print("Warning: Sample batch is None, skipping initialization")
            else:
                sample_volumes = sample_batch['volume']
                sample_masks = sample_batch['mask']
                
                # Process and stack tensors properly
                if isinstance(sample_volumes, list):
                    # Check if the volumes are already 5D tensors
                    if len(sample_volumes) > 0 and sample_volumes[0].dim() == 5:
                        sample_volumes = torch.cat(sample_volumes, dim=0)
                    else:
                        sample_volumes = torch.stack(sample_volumes)
                        
                if isinstance(sample_masks, list):
                    # Check if the masks are already 5D tensors
                    if len(sample_masks) > 0 and sample_masks[0].dim() == 5:
                        sample_masks = torch.cat(sample_masks, dim=0)
                    else:
                        sample_masks = torch.stack(sample_masks)
                
                # If tensors have extra dimensions, remove them
                if sample_volumes.dim() > 5:
                    sample_volumes = sample_volumes.squeeze(0)
                if sample_masks.dim() > 5:
                    sample_masks = sample_masks.squeeze(0)
                    
                # Move to device
                sample_volumes = sample_volumes.to(device)
                sample_masks = sample_masks.to(device)
                
                # Run a forward pass to initialize the classifier
                with torch.no_grad():
                    if use_mask_attention:
                        _ = model(sample_volumes, sample_masks)
                    else:
                        _ = model(sample_volumes)
                    
                print("Model classifier initialized successfully")
        except Exception as e:
            print(f"Error initializing model classifier: {e}")
            
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Validating')):
            try:
                # Handle case where batch could be None
                if batch is None:
                    print(f"Warning: Validation batch {batch_idx} is None, skipping")
                    continue
                    
                # Get data
                volumes = batch['volume']
                masks = batch['mask'] 
                labels = batch['label']
                
                # Process tensors to ensure correct dimensions
                if isinstance(volumes, list):
                    if len(volumes) == 0:
                        print(f"Warning: Empty volumes list in validation batch {batch_idx}, skipping")
                        continue
                        
                    # Check if the volumes are already 5D tensors
                    if volumes[0].dim() == 5:
                        volumes = torch.cat(volumes, dim=0)
                    else:
                        volumes = torch.stack(volumes)
                        
                if isinstance(masks, list):
                    if len(masks) == 0:
                        print(f"Warning: Empty masks list in validation batch {batch_idx}, skipping")
                        continue
                        
                    # Check if the masks are already 5D tensors
                    if masks[0].dim() == 5:
                        masks = torch.cat(masks, dim=0)
                    else:
                        masks = torch.stack(masks)
                
                # If tensors have extra dimensions, remove them
                if volumes.dim() > 5:
                    volumes = volumes.squeeze(0)
                if masks.dim() > 5:
                    masks = masks.squeeze(0)
                    
                # Move data to device
                volumes = volumes.to(device)
                masks = masks.to(device)
                
                # Adjust labels if needed
                if adjust_labels and isinstance(labels, list):
                    labels = [l - 1 for l in labels]
                labels = torch.tensor(labels).to(device)
                
                # First batch for debugging
                if batch_idx == 0:
                    print_tensor_info(volumes, "Validation input volumes")
                    print_tensor_info(masks, "Validation input masks")
                    print(f"Validation labels: {labels}")
                
                # Forward pass with or without mask attention
                if use_mask_attention:
                    outputs = model(volumes, masks)
                else:
                    outputs = model(volumes)
                
                if batch_idx == 0:
                    print_tensor_info(outputs, "Validation outputs")
                    # Print softmax probabilities to debug classification issues
                    softmax_probs = torch.softmax(outputs, dim=1)
                    print(f"Softmax probabilities: {softmax_probs}")
                
                # Calculate loss with or without mask weighting
                if use_mask_weighted_loss:
                    loss = mask_weighted_loss(outputs, labels, masks, criterion)
                else:
                    loss = criterion(outputs, labels).mean() if criterion.reduction == 'none' else criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * volumes.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Save predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Save preview if requested
                if save_preview and batch_idx == 0 and epoch is not None and output_dir is not None:
                    save_volume_preview(volumes, masks, outputs, labels, predicted, f"epoch_{epoch+1}_batch_0_val")
                
            except Exception as e:
                print(f"ERROR in validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print detailed classification results
    print("\nClassification Details:")
    print(f"Labels: {all_labels}")
    print(f"Predictions: {all_preds}")
    print(f"Correct predictions: {val_correct} / {val_total}")
    
    confusion = confusion_matrix(all_labels, all_preds) if len(all_labels) > 0 else None
    if confusion is not None:
        print("\nConfusion Matrix:")
        print(confusion)
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0) if len(all_labels) > 0 else {"macro avg": {"f1-score": 0}}
    
    # Calculate metrics
    val_loss = val_loss / max(1, val_total)
    val_acc = val_correct / max(1, val_total)
    
    # Calculate F1 score (macro average)
    val_f1 = report['macro avg']['f1-score']
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'f1': val_f1,
        'predictions': all_preds,
        'labels': all_labels
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get transforms
    transform, mask_transform = get_transforms(args.target_size, args.use_augmentation, args.margin)
    
    print(f"Setting up data loaders...")
    print(f"IMPORTANT: Using raw unmodified data with NO normalization")
    
    # Load data using optimized lazy loading
    try:
        # Use the masked dataloader or regular dataloader based on user choice
        if args.use_masked_dataloader:
            print(f"Using mask-focused dataloader that extracts subvolumes only from masked areas")
            dataloader = get_masked_nuclei_dataloader(
                root_dir=args.data_dir,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,  # Set to 0 for debugging
                transform=transform,
                mask_transform=mask_transform,
                class_csv_path=args.class_csv,
                filter_by_class=args.class_id,
                ignore_unclassified=True,
                target_size=tuple(args.target_size),
                sample_percent=args.sample_percent,
                mask_threshold=args.mask_threshold,
                min_masked_ratio=args.min_masked_ratio,
                scan_step=args.scan_step,
                debug=True
            )
        else:
            print(f"Using standard dataloader")
            dataloader = get_nuclei_dataloader(
                root_dir=args.data_dir,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                transform=transform,
                mask_transform=mask_transform,
                class_csv_path=args.class_csv,
                filter_by_class=args.class_id,
                ignore_unclassified=True,
                target_size=tuple(args.target_size),
                sample_percent=args.sample_percent
            )
        
        # Check the actual size of the samples
        try:
            sample_batch = next(iter(dataloader))
            if sample_batch is not None:
                if isinstance(sample_batch['volume'], list) and len(sample_batch['volume']) > 0:
                    sample_vol = sample_batch['volume'][0]
                    actual_vol_shape = sample_vol.shape
                    print(f"Actual sample volume shape: {actual_vol_shape}")
                    print(f"FINAL VERIFICATION - Raw data range (should match original): min: {sample_vol.min().item():.4f}, max: {sample_vol.max().item():.4f}")
                    print(f"NO NORMALIZATION APPLIED - ORIGINAL DATA VALUES PRESERVED")
                    if actual_vol_shape[1:] != tuple(args.target_size):
                        print(f"Warning: Actual volume size {actual_vol_shape[1:]} differs from target size {tuple(args.target_size)}")
                elif isinstance(sample_batch['volume'], torch.Tensor):
                    actual_vol_shape = sample_batch['volume'].shape
                    print(f"Actual batch volume shape: {actual_vol_shape}")
                    print(f"FINAL VERIFICATION - Raw data range (should match original): min: {sample_batch['volume'].min().item():.4f}, max: {sample_batch['volume'].max().item():.4f}")
                    print(f"NO NORMALIZATION APPLIED - ORIGINAL DATA VALUES PRESERVED")
            else:
                print("Warning: Sample batch is None, could not check dimensions")
                
            # Add extra verification with numerical data stats
            if sample_batch is not None and 'volume' in sample_batch:
                vol_data = sample_batch['volume']
                if isinstance(vol_data, list) and len(vol_data) > 0:
                    vol_data = vol_data[0]
                elif isinstance(vol_data, torch.Tensor):
                    vol_data = vol_data[0] if vol_data.dim() > 4 else vol_data
                
                # Print detailed statistics
                print("\nDETAILED DATA STATISTICS (should match raw data):")
                print(f"  Min: {vol_data.min().item():.4f}")
                print(f"  Max: {vol_data.max().item():.4f}")
                print(f"  Mean: {vol_data.mean().item():.4f}")
                print(f"  Std: {vol_data.std().item():.4f}")
                print(f"  25th percentile: {torch.quantile(vol_data.float().view(-1), 0.25).item():.4f}")
                print(f"  50th percentile (median): {torch.quantile(vol_data.float().view(-1), 0.5).item():.4f}")
                print(f"  75th percentile: {torch.quantile(vol_data.float().view(-1), 0.75).item():.4f}")
                
        except Exception as e:
            print(f"Unable to check sample dimensions: {e}")
        
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
        
        # Use the same collate function as the original dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataloader.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataloader.collate_fn
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
            
            # Handle class IDs that start from 1 instead of 0
            min_class = min(all_classes) if all_classes else 1
            max_class = max(all_classes) if all_classes else 19
            
            if min_class == 1:
                # Class IDs start from 1, so we need exactly max_class output classes
                args.output_classes = max_class
                print(f"Classes start from 1. Setting output classes to match max class ID: {args.output_classes}")
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
            
        # Create the masked adapter model
        model = MaskedVgg3DAdapter(
            base_model=base_model,
            input_size=tuple(args.target_size),
            num_classes=args.output_classes,
            classifier_size=512
        )
        
        if args.freeze_features:
            # Freeze feature extractor layers
            print("Freezing feature extractor layers")
            for param in model.features.parameters():
                param.requires_grad = False
                
            # Optionally unfreeze the last N convolutional blocks
            if args.unfreeze_last_n_layers != 0:
                print(f"Unfreezing the last {args.unfreeze_last_n_layers} convolutional blocks")
                
                # Get all convolutional layers
                conv_layers = []
                for i, layer in enumerate(model.features):
                    if isinstance(layer, nn.Conv3d):
                        conv_layers.append(i)
                
                # Determine how many convolution blocks to unfreeze
                total_blocks = len(conv_layers)
                
                if args.unfreeze_last_n_layers == -1:
                    # Unfreeze all blocks
                    start_idx = 0
                    print(f"Unfreezing all {total_blocks} convolutional blocks")
                else:
                    # Unfreeze only the last N blocks
                    blocks_to_unfreeze = min(args.unfreeze_last_n_layers, total_blocks)
                    start_idx = max(0, len(conv_layers) - blocks_to_unfreeze)
                    print(f"Unfreezing the last {blocks_to_unfreeze} of {total_blocks} convolutional blocks")
                
                # Unfreeze parameters in the last N blocks
                for i in range(start_idx, len(conv_layers)):
                    conv_idx = conv_layers[i]
                    # Unfreeze this conv layer
                    for param in model.features[conv_idx].parameters():
                        param.requires_grad = True
                    
                    # Also unfreeze the corresponding BatchNorm (usually right after Conv)
                    if conv_idx + 1 < len(model.features) and isinstance(model.features[conv_idx + 1], nn.BatchNorm3d):
                        for param in model.features[conv_idx + 1].parameters():
                            param.requires_grad = True
                
                # Count trainable parameters
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        
        # Set up optimizer
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(
                [p for p in model.parameters() if p.requires_grad], 
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        else:  # adam
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad], 
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        
        # Train the model
        print("\nStarting training...")
        model, history = train_model(model, train_loader, val_loader, optimizer, args)
        
        # Plot training history
        history_plot_path = os.path.join(args.output_dir, 'masked_training_history.png')
        plot_training_history(history, history_plot_path)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, 'final_masked_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
        
    except Exception as e:
        print(f"Error in training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Utility functions inherited from finetune_vgg3d.py
def get_transforms(target_size, use_augmentation=False, margin=0):
    """Define transformations for the input data."""
    # This function should be imported from the original file or defined here
    if use_augmentation:
        return partial(augmented_transform, margin=margin), mask_transform
    else:
        return partial(basic_transform, margin=margin), mask_transform

def basic_transform(x, margin=0):
    """Transform a 3D volume to a 5D tensor with shape (1, 1, D, H, W) with ABSOLUTELY NO NORMALIZATION"""
    # Convert to numpy array if not already
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # Apply margin if specified
    if margin > 0 and x.ndim == 3:
        safe_margin = min(margin, min(x.shape) // 4)
        if safe_margin > 0:
            x = x[safe_margin:-safe_margin, safe_margin:-safe_margin, safe_margin:-safe_margin]
    
    # Convert to tensor without ANY normalization - keep raw values exactly as they are
    x_tensor = torch.from_numpy(x.astype(np.float32))
    
    # Print raw stats for debugging
    # print(f"RAW DATA VALUES - min: {x.min()}, max: {x.max()}, mean: {x.mean():.2f}, std: {x.std():.2f}")
    
    # Reshape to expected format (1, 1, D, H, W)
    if x_tensor.dim() == 3:  # (D, H, W)
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
    elif x_tensor.dim() == 4:  # (C, D, H, W) or (B, D, H, W)
        x_tensor = x_tensor.unsqueeze(0)  # -> (1, C, D, H, W)
    
    return x_tensor

def mask_transform(x):
    """Transform a mask to a 5D tensor with shape (1, 1, D, H, W) - preserving raw values"""
    if x is None:
        return None
    
    # Convert to tensor without any normalization
    x_tensor = torch.from_numpy(x.astype(np.float32))
    
    # Reshape to expected format (1, 1, D, H, W)
    if x_tensor.dim() == 3:  # (D, H, W)
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
    
    return x_tensor

def augmented_transform(x, margin=0):
    """Transform with augmentations (simplified from original)"""
    # First apply basic transform (no normalization)
    x_tensor = basic_transform(x, margin)
    
    # Apply basic augmentations (flips, rotations)
    if np.random.rand() < 0.5:  # Random flip along z-axis
        x_tensor = torch.flip(x_tensor, dims=[2])
    if np.random.rand() < 0.5:  # Random flip along y-axis
        x_tensor = torch.flip(x_tensor, dims=[3])
    if np.random.rand() < 0.5:  # Random flip along x-axis
        x_tensor = torch.flip(x_tensor, dims=[4])
    
    return x_tensor

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

if __name__ == "__main__":
    main() 