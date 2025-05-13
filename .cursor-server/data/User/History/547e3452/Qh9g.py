import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
from scipy.ndimage import zoom

class TiffVolumeDataset(Dataset):
    """Dataset for loading 3D volumes from sequences of TIFF files."""
    
    def __init__(self, root_dir, input_dir="raw", target_dir="mask", transform=None, target_size=(104, 104, 104)):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory containing sample folders
            input_dir (str): Name of subdirectory containing raw TIFF slices
            target_dir (str): Name of subdirectory containing mask TIFF slices
            transform (callable, optional): Optional transform to apply to samples
            target_size (tuple): Target size for all volumes (H, W, D)
        """
        self.root_dir = root_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_size = target_size
        
        # Check if root_dir is directly a sample directory or contains sample directories
        if os.path.isdir(os.path.join(root_dir, input_dir)) and os.path.isdir(os.path.join(root_dir, target_dir)):
            # Root directory is a sample directory
            self.sample_dirs = [root_dir]
        else:
            # Root directory contains multiple sample directories
            self.sample_dirs = []
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    input_path = os.path.join(item_path, input_dir)
                    target_path = os.path.join(item_path, target_dir)
                    if os.path.isdir(input_path) and os.path.isdir(target_path):
                        self.sample_dirs.append(item_path)
        
        print(f"Found {len(self.sample_dirs)} sample directories")
        
    def __len__(self):
        return len(self.sample_dirs)
    
    def load_tiff_sequence(self, directory):
        """Load a sequence of TIFF files into a 3D volume."""
        tiff_files = sorted(glob.glob(os.path.join(directory, "*.tif*")))
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {directory}")
        
        # Load all slices
        slices = []
        for tiff_file in tiff_files:
            slice_data = tifffile.imread(tiff_file)
            slices.append(slice_data)
        
        # Stack slices into volume
        volume = np.stack(slices, axis=0)
        
        # Handle dimensions
        if volume.ndim == 3:
            # Add channel dimension if missing
            volume = volume[np.newaxis, ...]
        elif volume.ndim == 4:
            # Rearrange dimensions (Z,C,Y,X) to (C,Z,Y,X)
            volume = np.transpose(volume, (1, 0, 2, 3))
        
        return volume
        
    def resize_volume(self, volume, target_size):
        """Resize a volume to the target size."""
        # Get current size (excluding channel dimension)
        c, d, h, w = volume.shape
        
        # Calculate zoom factors for each dimension
        # We don't resize the channel dimension
        factors = [1, target_size[2] / d, target_size[0] / h, target_size[1] / w]
        
        # Resize using scipy zoom
        resized = np.zeros((c,) + target_size[::-1], dtype=volume.dtype)
        
        for i in range(c):
            # Resize each channel separately
            resized[i] = zoom(volume[i], factors[1:], order=1)
            
        return resized
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # Load input volume
        input_path = os.path.join(sample_dir, self.input_dir)
        try:
            input_volume = self.load_tiff_sequence(input_path)
            input_volume = self.resize_volume(input_volume, self.target_size)
        except Exception as e:
            print(f"Error loading/resizing input from {input_path}: {str(e)}")
            # Create dummy data for error cases
            input_volume = np.zeros((1,) + self.target_size[::-1], dtype=np.float32)
        
        # Load target volume
        target_path = os.path.join(sample_dir, self.target_dir)
        try:
            target_volume = self.load_tiff_sequence(target_path)
            target_volume = self.resize_volume(target_volume, self.target_size)
        except Exception as e:
            print(f"Error loading/resizing target from {target_path}: {str(e)}")
            # Create dummy data for error cases
            target_volume = np.zeros((1,) + self.target_size[::-1], dtype=np.float32)
        
        # Normalize to [0, 1] range
        input_volume = input_volume.astype(np.float32)
        if input_volume.max() > 0:
            input_volume = input_volume / input_volume.max()
            
        target_volume = target_volume.astype(np.float32)
        if target_volume.max() > 0:
            target_volume = target_volume / target_volume.max()
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_volume)
        target_tensor = torch.from_numpy(target_volume)
        
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        
        return {'input': input_tensor, 'target': target_tensor}


def get_tiff_dataloader(root_dir, batch_size=4, shuffle=True, num_workers=0, 
                        input_dir="raw", target_dir="mask", transform=None, target_size=(104, 104, 104)):
    """
    Create a DataLoader for loading TIFF volume data.
    
    Args:
        root_dir (str): Directory containing sample folders
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes for DataLoader
        input_dir (str): Name of subdirectory containing raw TIFF slices
        target_dir (str): Name of subdirectory containing mask TIFF slices
        transform (callable, optional): Optional transform to apply to samples
        target_size (tuple): Target size for all volumes (H, W, D)
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = TiffVolumeDataset(
        root_dir=root_dir,
        input_dir=input_dir,
        target_dir=target_dir,
        transform=transform,
        target_size=target_size
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader 