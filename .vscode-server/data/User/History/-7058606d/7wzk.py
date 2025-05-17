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

# def load_chromatin_classes(csv_path, ignore_unclassified=True):
#     """
#     Load chromatin class information from CSV file.
    
#     Args:
#         csv_path (str): Path to the CSV file containing chromatin class information.
#         ignore_unclassified (bool): Whether to ignore entries with class_name "Unclassified".
        
#     Returns:
#         dict: Dictionary mapping sample_id to class information (id and name).
#     """
#     if not os.path.exists(csv_path):
#         print(f"Warning: Chromatin class CSV file not found: {csv_path}")
#         return {}
    
#     df = pd.read_csv(csv_path)
    
#     sample_to_class = {}
#     for _, row in df.iterrows():
#         if ignore_unclassified and (row['class_name'] == 'Unclassified' or row['class_id'] == 19):
#             continue
#         sample_to_class[str(row['sample_id'])] = {
#             'class_id': row['class_id'],
#             'class_name': row['class_name']
#         }
    
#     return sample_to_class


def default_transform(image):
    """
    Default transform for 3D volume processing (no resizing).
    
    Args:
        image (numpy.ndarray): Input volume of shape [Z, H, W] or [H, W]
        
    Returns:
        torch.Tensor: Processed volume tensor of shape [Z, 1, H, W] or [1, H, W]
    """
    if len(image.shape) == 3:  # If it's a 3D volume
        transformed_slices = []
        for z in range(image.shape[0]):
            slice_img = Image.fromarray(image[z].astype(np.uint8))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            transformed_slices.append(transform(slice_img))
        return torch.stack(transformed_slices, dim=0)  # Stack along first dimension
    else:  # If it's a 2D image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        return transform(image)


class LowResImageDataset(Dataset):
    """
    Dataset for loading low-resolution raw images to capture coarse texture.
    """
    def __init__(self, 
                 root_dir,
                 transform=None,
                 target_size=(64, 64),
                 sample_percent=100,
                 sample_ids=None,
                 z_window_size=80 , # Now represents desired number of frames
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
            z_window_size (int): Desired number of frames in output.
            debug (bool): Whether to print debug information.
        """
        self.root_dir = root_dir
        self.transform = transform if transform else lambda x: default_transform(x, target_size)
        self.target_size = target_size
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)
        self.z_window_size = z_window_size
        
        self.samples = []
        
        # Directory-based loading
        print(f"Loading samples from directory structure: {root_dir}")
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_id_map = {name: idx for idx, name in enumerate(sorted(class_dirs))}
        for class_name in class_dirs:
            
            
            class_path = os.path.join(root_dir, class_name)
            sample_names = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            if sample_ids is not None:
                sample_names = [s for s in sample_names if s in sample_ids or s.zfill(4) in sample_ids]
            if self.sample_percent < 100:
                n_keep = max(1, int(len(sample_names) * self.sample_percent / 100.0))
                sample_names = random.sample(sample_names, n_keep)
            for sample_name in sample_names:
                sample_dir = os.path.join(class_path, sample_name)
                raw_dir = os.path.join(sample_dir, 'raw')
                if not os.path.exists(raw_dir):
                    if self.debug:
                        print(f"Warning: Raw directory not found for sample {sample_name} in class {class_name}")
                    continue
                raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
                if not raw_files:
                    if self.debug:
                        print(f"Warning: No raw files found for sample {sample_name} in class {class_name}")
                    continue
                self.samples.append({
                    'sample_id': sample_name,
                    'class_id': class_id_map[class_name],
                    'class_name': class_name,
                    'window_files': raw_files,
                    'window_desc': f"0-{len(raw_files)-1}"
                })
        print(f"Total number of samples to process: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_and_average_window(self, window_files):
        """
        Load a set of image files to create a 3D volume, dynamically selecting frames
        to match the desired z_window_size.
        
        Args:
            window_files (list): List of file paths to load
            
        Returns:
            numpy.ndarray: 3D volume of shape [z_window_size, H, W]
        """
        if len(window_files) == 0:
            return np.zeros((self.z_window_size, 64, 64), dtype=np.float32)
        
        # Calculate stride to get exactly z_window_size frames
        total_frames = len(window_files)
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
        for idx in indices:
            try:
                img = np.array(Image.open(window_files[idx]), dtype=np.float32)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {window_files[idx]}: {e}")
                # On error, append a blank frame
                images.append(np.zeros_like(images[0]) if images else np.zeros((64, 64), dtype=np.float32))
        
        # Stack images along first dimension to create 3D volume
        volume = np.stack(images, axis=0)
        return volume
    
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
            window_files = sample['window_files']
            
            
            try:
                avg_image = self._load_and_average_window(window_files)
                image_tensor = self.transform(avg_image)
            except Exception as e:
                print(f"Error processing window for sample {sample_id}: {e}")
                
                image_tensor = torch.zeros((1, *self.target_size), dtype=torch.float32)
            
            
            metadata = {
                'sample_id': sample_id,
                'class_name': sample['class_name'],
                'window_desc': sample['window_desc']
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


def get_lowres_image_dataloader(root_dir,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                transform=None,
                                sample_ids=None,
                                class_csv_path=None,
                                filter_by_class=None,
                                ignore_unclassified=True,
                                # target_size=(64, 64), 
                                target_size=(80, 80),
                                sample_percent=100,
                                z_window_size=80,
                                pin_memory=False,
                                debug=False):
    """
    Create a DataLoader for low-resolution images with coarse texture.
    
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
        z_window_size (int): Desired number of frames in output.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        debug (bool): Whether to print debug information.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the low-res image dataset.
    """
    dataset = LowResImageDataset(
        root_dir=root_dir,
        transform=transform,
        sample_ids=sample_ids,
        
        
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
    # Use the default_transform by not passing a custom transform
    dataloader = get_lowres_image_dataloader(
        root_dir="low_res_dataset",
        batch_size=16,
        z_window_size=80,
        debug=True
    )
    print(f"Dataset size: {len(dataloader.dataset)}")
    for batch in dataloader:
        print(f"Batch size: {len(batch['label'])}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Sample IDs: {batch['metadata']['sample_id']}")
        break