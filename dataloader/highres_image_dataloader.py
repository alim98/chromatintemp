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

def load_chromatin_classes(csv_path, ignore_unclassified=True):
    """
    Load chromatin class information from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing chromatin class information.
        ignore_unclassified (bool): Whether to ignore entries with class_name "Unclassified".
        
    Returns:
        dict: Dictionary mapping sample_id to class information (id and name).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Chromatin class CSV file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    
    sample_to_class = {}
    for _, row in df.iterrows():
        if ignore_unclassified and (row['class_name'] == 'Unclassified' or row['class_id'] == 19):
            continue
        sample_to_class[str(row['sample_id'])] = {
            'class_id': row['class_id'],
            'class_name': row['class_name']
        }
    
    return sample_to_class


def default_transform(image, target_size=(224, 224)):
    """
    Default transform for 3D volume processing.
    
    Args:
        image (numpy.ndarray): Input volume of shape [Z, H, W] or a single image
        target_size (tuple): Target size (height, width)
        
    Returns:
        torch.Tensor: Processed volume tensor of shape [Z, 1, H, W] or [1, H, W]
    """
    if len(image.shape) == 3:  # If it's a 3D volume
        transformed_slices = []
        for z in range(image.shape[0]):
            slice_img = Image.fromarray(image[z].astype(np.uint8))
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            transformed_slices.append(transform(slice_img))
        return torch.stack(transformed_slices, dim=0)  # Stack along first dimension
    else:  # If it's a 2D image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        return transform(image)


class HighResImageDataset(Dataset):
    """
    Dataset for loading high-resolution raw images with full 3D volume support.
    """
    def __init__(self, 
                 root_dir,
                 transform=None,
                 sample_ids=None,
                 class_csv_path=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 target_size=(224, 224),
                 sample_percent=100,
                 z_window_size=80,  # Desired number of slices in output volume
                 debug=False):
        """
        Args:
            root_dir (str): Root directory containing the samples.
            transform (callable, optional): Transform to apply to the images.
            sample_ids (list, optional): List of sample IDs to include.
            class_csv_path (str, optional): Path to CSV file with class information.
            filter_by_class (int or list, optional): Class ID(s) to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            target_size (tuple): Target image size (height, width).
            sample_percent (int): Percentage of samples to load per class (1-100).
            z_window_size (int): Desired number of slices in output volume.
            debug (bool): Whether to print debug information.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else lambda x: default_transform(x, target_size)
        self.target_size = target_size
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)
        self.z_window_size = z_window_size
        
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
                        'file_paths': raw_files,  # All image files for this sample
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
    
    def _load_volume(self, file_paths):
        """
        Load a set of image files to create a 3D volume, dynamically selecting frames
        to match the desired z_window_size.
        
        Args:
            file_paths (list): List of file paths to load
                
        Returns:
            numpy.ndarray: 3D volume of shape [z_window_size, H, W]
        """
        if len(file_paths) == 0:
            return np.zeros((self.z_window_size, 224, 224), dtype=np.float32)
        
        # Calculate stride to get exactly z_window_size frames
        total_frames = len(file_paths)
        if total_frames <= self.z_window_size:
            # If we have fewer frames than needed, use all and pad
            indices = list(range(total_frames))
            # Pad indices by repeating the last frame
            indices.extend([indices[-1]] * (self.z_window_size - total_frames))
        else:
            # Calculate indices to get exactly z_window_size frames
            stride = (total_frames - 1) / (self.z_window_size - 1)
            indices = [int(i * stride) for i in range(self.z_window_size)]
        
        images = []
        slice_indices = []
        for idx in indices:
            try:
                img = np.array(Image.open(file_paths[idx]), dtype=np.float32)
                images.append(img)
                slice_idx = os.path.splitext(os.path.basename(file_paths[idx]))[0]
                slice_indices.append(slice_idx)
            except Exception as e:
                print(f"Error loading image {file_paths[idx]}: {e}")
                # On error, append a blank frame
                images.append(np.zeros_like(images[0]) if images else np.zeros((224, 224), dtype=np.float32))
                slice_indices.append("error")
        
        # Stack images along first dimension to create 3D volume
        volume = np.stack(images, axis=0)
        return volume, slice_indices
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample dictionary with image, label, and metadata
        """
        try:
            sample = self.samples[idx]
            sample_id = sample['sample_id']
            file_paths = sample['file_paths']
            
            try:
                volume, slice_indices = self._load_volume(file_paths)
                image_tensor = self.transform(volume)
            except Exception as e:
                print(f"Error processing volume for sample {sample_id}: {e}")
                image_tensor = torch.zeros((self.z_window_size, 1, *self.target_size), dtype=torch.float32)
                slice_indices = ["error"] * self.z_window_size
            
            metadata = {
                'sample_id': sample_id,
                'class_name': sample['class_name'],
                'slice_indices': slice_indices,
                'total_slices': sample['total_slices']
            }
            
            return {
                'image': image_tensor,
                'label': sample['class_id'],
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx={idx}: {e}")
            import traceback
            traceback.print_exc()
            raise


def get_highres_image_dataloader(root_dir,
                                 batch_size=16,
                                 shuffle=True,
                                 num_workers=4,
                                 transform=None,
                                 sample_ids=None,
                                 class_csv_path=None,
                                 filter_by_class=None,
                                 ignore_unclassified=True,
                                 target_size=(224, 224),
                                 sample_percent=100,
                                 z_window_size=80,
                                 pin_memory=False,
                                 debug=False):
    """
    Create a DataLoader for high-resolution images.
    
    Args:
        root_dir (str): Root directory containing the samples.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes.
        transform (callable, optional): Transform to apply to the images.
        sample_ids (list, optional): List of sample IDs to include.
        class_csv_path (str, optional): Path to CSV file with class information.
        filter_by_class (int or list, optional): Class ID(s) to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        target_size (tuple): Target image size (height, width).
        sample_percent (int): Percentage of samples to load per class (1-100).
        z_window_size (int): Desired number of slices in output volume.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        debug (bool): Whether to print debug information.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the high-res image dataset.
    """
    dataset = HighResImageDataset(
        root_dir=root_dir,
        transform=transform,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        target_size=target_size,
        sample_percent=sample_percent,
        z_window_size=z_window_size,
        debug=debug
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataloader = get_highres_image_dataloader(
        root_dir="data/nuclei_sample_1a_v1",
        batch_size=8,
        class_csv_path="chromatin_classes_and_samples.csv",
        transform=transform,
        z_window_size=3,
        debug=True
    )
    
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    
    for batch in dataloader:
        print(f"Batch size: {len(batch['label'])}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Sample IDs: {batch['metadata']['sample_id']}")
        break 