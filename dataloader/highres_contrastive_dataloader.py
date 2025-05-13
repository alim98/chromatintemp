import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.contrastive_transforms import get_transforms_for_contrastive, collate_contrastive


class HighResContrastiveDataset(Dataset):
    """
    Dataset for high-resolution images with contrastive learning support,
    compatible with MorphoFeatures texture models.
    """
    def __init__(self, 
                 root_dir,
                 transforms=None,
                 transforms_sim=None,
                 sample_ids=None,
                 class_csv_path=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 target_size=(224, 224),
                 sample_percent=100,
                 z_window_size=5,
                 predict=False,
                 debug=False):
        """
        Args:
            root_dir (str): Root directory containing the samples.
            transforms (callable, optional): Base transform for target images.
            transforms_sim (callable, optional): Transform for creating augmented views.
            sample_ids (list, optional): List of sample IDs to include.
            class_csv_path (str, optional): Path to CSV file with class information.
            filter_by_class (int or list, optional): Class ID(s) to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            target_size (tuple): Target image size (height, width).
            sample_percent (int): Percentage of samples to load per class (1-100).
            z_window_size (int): Number of z-slices to use per sample.
            predict (bool): Whether to use the dataset for prediction (no pairs).
            debug (bool): Whether to print debug information.
        """
        self.root_dir = root_dir
        
        # If transforms not provided, use defaults
        if transforms is None or transforms_sim is None:
            self.transforms, self.transforms_sim = get_transforms_for_contrastive(target_size)
        else:
            self.transforms = transforms
            self.transforms_sim = transforms_sim
            
        self.target_size = target_size
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)
        self.z_window_size = z_window_size
        self.predict = predict
        
        self.samples = []
        
        if class_csv_path and os.path.exists(class_csv_path):
            try:
                df = pd.read_csv(class_csv_path)
                
                if ignore_unclassified:
                    df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
                
                if filter_by_class is not None:
                    if isinstance(filter_by_class, int):
                        filter_by_class = [filter_by_class]
                    df = df[df['class_id'].isin(filter_by_class)]
                
                if sample_ids is not None:
                    sample_ids = [str(sid).zfill(4) for sid in sample_ids]
                    df = df[df['sample_id'].astype(str).apply(lambda x: x.zfill(4)).isin(sample_ids)]
                
                if self.sample_percent < 100:
                    sampled_df = pd.DataFrame()
                    for class_id, group in df.groupby('class_id'):
                        num_to_keep = max(1, int(len(group) * self.sample_percent / 100.0))
                        sampled_group = group.sample(n=num_to_keep, random_state=42)
                        sampled_df = pd.concat([sampled_df, sampled_group])
                        if self.debug:
                            print(f"Class {class_id}: Sampled {num_to_keep}/{len(group)} samples ({self.sample_percent}%)")
                    df = sampled_df
                
                for _, row in df.iterrows():
                    sample_id = str(row['sample_id']).zfill(4)
                    class_id = int(row['class_id'])
                    class_name = row['class_name']
                    
                    if ignore_unclassified and (class_name == 'Unclassified' or class_id == 19):
                        continue
                    
                    padded_dir = os.path.join(root_dir, sample_id)
                    unpadded_dir = os.path.join(root_dir, sample_id.lstrip('0'))
                    
                    sample_dir = None
                    if os.path.isdir(padded_dir):
                        sample_dir = padded_dir
                    elif os.path.isdir(unpadded_dir) and sample_id.lstrip('0') != '':
                        sample_dir = unpadded_dir
                    
                    if sample_dir is None:
                        print(f"Warning: Sample directory not found for {sample_id}")
                        continue
                    
                    raw_dir = os.path.join(sample_dir, 'raw')
                    if not os.path.exists(raw_dir):
                        print(f"Warning: Raw directory not found for sample {sample_id}")
                        continue
                    
                    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
                    if not raw_files:
                        print(f"Warning: No raw files found for sample {sample_id}")
                        continue
                    
                    self.samples.append({
                        'sample_id': sample_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'file_paths': raw_files,
                        'total_slices': len(raw_files)
                    })
                
                print(f"Total number of samples to process: {len(self.samples)}")
                
            except Exception as e:
                print(f"Error loading class CSV: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Class CSV file not provided or not found: {class_csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_volume_slices(self, file_paths):
        """
        Load a subset of slices from the volume for contrastive learning.
        
        Args:
            file_paths (list): List of file paths to load
                
        Returns:
            numpy.ndarray: Selected slices from the volume
        """
        if len(file_paths) == 0:
            return np.zeros((self.z_window_size, 224, 224), dtype=np.float32)
        
        total_frames = len(file_paths)
        
        if total_frames <= self.z_window_size:
            # If we have fewer frames than needed, use all and pad
            selected_indices = list(range(total_frames))
            # Pad indices by repeating the last frame
            selected_indices.extend([selected_indices[-1]] * (self.z_window_size - total_frames))
        else:
            # Select a random central region
            start_idx = random.randint(0, total_frames - self.z_window_size)
            selected_indices = list(range(start_idx, start_idx + self.z_window_size))
        
        images = []
        for idx in selected_indices:
            try:
                img = np.array(Image.open(file_paths[idx]), dtype=np.float32)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {file_paths[idx]}: {e}")
                # On error, append a blank frame
                images.append(np.zeros_like(images[0]) if images else np.zeros((224, 224), dtype=np.float32))
        
        # Stack images along first dimension to create 3D volume
        volume = np.stack(images, axis=0)
        return volume
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index
            
        Returns:
            If predict=True:
                torch.Tensor: Processed volume tensor
            Else:
                tuple: (input_tensor_pair, target_tensor_pair) for contrastive learning
        """
        try:
            sample = self.samples[idx]
            sample_id = sample['sample_id']
            file_paths = sample['file_paths']
            
            # Load volume slices
            volume = self._load_volume_slices(file_paths)
            
            if self.predict:
                # For prediction, just return the processed volume
                return self.transforms(volume)
            
            # For training, create two versions of the data for contrastive learning
            target_views = [self.transforms(volume.copy()) for _ in range(2)]
            input_views = [self.transforms_sim(view.clone()) for view in target_views]
            
            return torch.stack(input_views), torch.stack(target_views)
            
        except Exception as e:
            print(f"Error in __getitem__ for idx={idx}: {e}")
            import traceback
            traceback.print_exc()
            raise


def get_highres_contrastive_loaders(root_dir,
                               batch_size=16,
                               shuffle=True,
                               num_workers=4,
                               transforms=None,
                               transforms_sim=None,
                               sample_ids=None,
                               class_csv_path=None,
                               filter_by_class=None,
                               ignore_unclassified=True,
                               target_size=(224, 224),
                               sample_percent=100,
                               z_window_size=5,
                               validation_split=0.2,
                               seed=42,
                               pin_memory=False,
                               debug=False):
    """
    Create train and validation DataLoaders for high-resolution contrastive learning.
    
    Args:
        root_dir (str): Root directory containing the samples.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes.
        transforms (callable, optional): Base transform for target images.
        transforms_sim (callable, optional): Transform for creating augmented views.
        sample_ids (list, optional): List of sample IDs to include.
        class_csv_path (str, optional): Path to CSV file with class information.
        filter_by_class (int or list, optional): Class ID(s) to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        target_size (tuple): Target image size (height, width).
        sample_percent (int): Percentage of samples to load per class (1-100).
        z_window_size (int): Number of z-slices to use per sample.
        validation_split (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        debug (bool): Whether to print debug information.
        
    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    # Create the full dataset
    full_dataset = HighResContrastiveDataset(
        root_dir=root_dir,
        transforms=transforms,
        transforms_sim=transforms_sim,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        target_size=target_size,
        sample_percent=sample_percent,
        z_window_size=z_window_size,
        predict=False,
        debug=debug
    )
    
    # Split into train and validation
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(np.floor(validation_split * num_samples))
    
    # Set seed for reproducibility
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_contrastive,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_contrastive,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == "__main__":
    # Example usage
    loaders = get_highres_contrastive_loaders(
        root_dir="data/nuclei_sample_1a_v1",
        batch_size=8,
        class_csv_path="chromatin_classes_and_samples.csv",
        z_window_size=3,
        debug=True
    )
    
    print(f"Train dataset size: {len(loaders['train'].dataset)}")
    print(f"Val dataset size: {len(loaders['val'].dataset)}")
    
    # Test one batch
    for inputs, targets in loaders['train']:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape if isinstance(targets, torch.Tensor) else 'list of tensors'}")
        break 