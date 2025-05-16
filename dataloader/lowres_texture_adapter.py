#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import logging
import tifffile
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Configure logging
logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s] %(message)s',
                   stream=sys.stdout,
                   level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to path to allow imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.lowres_image_dataloader import get_lowres_image_dataloader, LowResImageDataset

def adapt_batch_for_morphofeatures_texture(batch):
    """
    Adapt a single batch for MorphoFeatures texture model.
    
    Args:
        batch: A batch dictionary from the lowres dataloader
        
    Returns:
        List of tensors that the Inferno trainer can process
    """
    # Process image tensor - should be [B, Z, 1, H, W] from dataloader
    if 'image' in batch:
        images = batch['image']  # Shape: [B, Z, 1, H, W]
        
        # Reshape to [B, 1, Z, H, W] for 3D UNet
        volumes = images.transpose(1, 2)
        
        # For Inferno trainer, return input and target tensors directly 
        # (not in a dictionary)
        return [volumes, volumes]  # Input, target pairs
    
    # If no images, return empty tensors
    return [torch.tensor([]), torch.tensor([])]

def adapted_collate_fn(original_collate_fn, batch):
    """
    Adapter for the collate function that can be pickled.
    
    Args:
        original_collate_fn: Original collate function
        batch: Batch to process
        
    Returns:
        Adapted batch
    """
    original_batch = original_collate_fn(batch)
    return adapt_batch_for_morphofeatures_texture(original_batch)

def adapt_lowres_dataloader_for_morphofeatures(dataloader):
    """
    Adapt the lowres dataloader for MorphoFeatures texture model.
    
    Args:
        dataloader: DataLoader instance from get_lowres_image_dataloader
        
    Returns:
        DataLoader: New DataLoader with adapted collate function
    """
    # Create a new dataloader with adapted collate function
    original_collate_fn = dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else torch.utils.data.dataloader.default_collate
    
    # Use a functools.partial to create a new function with the original collate_fn bound
    from functools import partial
    adapted_fn = partial(adapted_collate_fn, original_collate_fn)
    
    # Clone the dataloader with new collate function
    new_dataloader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler),
        num_workers=dataloader.num_workers,
        collate_fn=adapted_fn,
        pin_memory=dataloader.pin_memory,
    )
    
    return new_dataloader

def custom_transform_for_texture(image, target_size=(104, 104)):
    """
    Simple transform function for 3D volume processing without using torchvision.
    
    Args:
        image (numpy.ndarray): Input volume of shape [Z, H, W]
        target_size (tuple): Target size (height, width)
        
    Returns:
        torch.Tensor: Processed volume tensor of shape [Z, 1, H, W]
    """
    if len(image.shape) == 3:  # If it's a 3D volume
        # Center-crop or pad the z-dimension if needed
        z_dim = image.shape[0]
        target_z = target_size[0]  # Assuming cubic box where target_size[0] == target_size[2]
        
        if z_dim > target_z:
            # Center crop
            start_z = (z_dim - target_z) // 2
            image = image[start_z:start_z+target_z]
        elif z_dim < target_z:
            # Pad
            pad_before = (target_z - z_dim) // 2
            pad_after = target_z - z_dim - pad_before
            padded_image = np.zeros((target_z, image.shape[1], image.shape[2]), dtype=image.dtype)
            padded_image[pad_before:pad_before+z_dim] = image
            image = padded_image
        
        # Process each slice
        transformed_slices = []
        for z in range(image.shape[0]):
            slice_img = image[z].astype(np.uint8)
            
            # Center crop or pad each slice
            h, w = slice_img.shape
            if h > target_size[0] or w > target_size[1]:
                # Center crop
                start_h = max(0, (h - target_size[0]) // 2)
                start_w = max(0, (w - target_size[1]) // 2)
                slice_img = slice_img[start_h:start_h+target_size[0], start_w:start_w+target_size[1]]
            elif h < target_size[0] or w < target_size[1]:
                # Pad with zeros
                new_img = np.zeros(target_size, dtype=slice_img.dtype)
                start_h = max(0, (target_size[0] - h) // 2)
                start_w = max(0, (target_size[1] - w) // 2)
                new_img[start_h:start_h+h, start_w:start_w+w] = slice_img
                slice_img = new_img
            
            # Convert to PIL for resizing if needed
            pil_img = Image.fromarray(slice_img)
            if pil_img.size != target_size:
                pil_img = pil_img.resize(target_size, Image.BICUBIC)
            
            # Convert back to numpy, normalize, and convert to tensor
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            # Normalize to [-1, 1] range (similar to normalize with mean=0.5, std=0.5)
            img_np = (img_np - 0.5) / 0.5
            
            # Create tensor and add channel dimension
            tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add channel dimension
            transformed_slices.append(tensor)
        
        # Stack along z dimension
        volume = torch.stack(transformed_slices, dim=0)
        return volume
    else:
        raise ValueError(f"Expected 3D input volume, got shape: {image.shape}")

def make_transform_function(target_size):
    """
    Create a transform function for the given target size.
    Returns a proper named function instead of a lambda to fix pickling issues.
    
    Args:
        target_size (tuple): Target size for the transform
        
    Returns:
        function: Transform function that can be pickled
    """
    def transform_func(image):
        return custom_transform_for_texture(image, target_size=target_size)
    return transform_func

class TiffVolumeDataset(Dataset):
    """Dataset for loading 3D volumes from stacks of TIFF images."""
    
    def __init__(self, sample_ids, is_cytoplasm=False, box_size=(104, 104, 104), 
                 input_dir="raw", target_dir="mask"):
        """
        Initialize the dataset.
        
        Args:
            sample_ids (list): List of directories containing raw and mask subdirectories with TIFF slices
            is_cytoplasm (bool): Whether we're processing cytoplasm data
            box_size (tuple): Size of the 3D patches to extract
            input_dir (str): Name of the directory containing input TIFF slices
            target_dir (str): Name of the directory containing target/mask TIFF slices
        """
        self.sample_ids = sample_ids
        self.is_cytoplasm = is_cytoplasm
        self.box_size = box_size
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        # Validate sample directories
        valid_samples = []
        for sample_path in sample_ids:
            raw_dir = os.path.join(sample_path, input_dir)
            mask_dir = os.path.join(sample_path, target_dir)
            
            if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.tif')])
                mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
                
                if raw_files and mask_files:
                    valid_samples.append(sample_path)
                    logger.info(f"Found valid sample: {sample_path} with {len(raw_files)} raw slices and {len(mask_files)} mask slices")
                else:
                    logger.warning(f"Sample {sample_path} has no TIFF files in {input_dir} or {target_dir}")
            else:
                logger.warning(f"Sample {sample_path} missing {input_dir} or {target_dir} directory")
        
        if not valid_samples:
            raise ValueError(f"No valid samples found among {len(sample_ids)} provided paths")
        
        self.valid_samples = valid_samples
        logger.info(f"Dataset initialized with {len(valid_samples)} valid samples")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """Get a 3D patch from the sample at index idx."""
        sample_path = self.valid_samples[idx]
        
        # Load raw and mask volumes from TIFF slices
        raw_vol = self._load_tiff_volume(os.path.join(sample_path, self.input_dir))
        mask_vol = self._load_tiff_volume(os.path.join(sample_path, self.target_dir))
        
        # Make sure volumes have the same shape
        if raw_vol.shape != mask_vol.shape:
            logger.warning(f"Raw and mask volumes have different shapes: {raw_vol.shape} vs {mask_vol.shape}")
            # Adjust sizes to be the same by cropping the larger one
            min_shape = [min(raw_vol.shape[i], mask_vol.shape[i]) for i in range(3)]
            raw_vol = raw_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask_vol = mask_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Extract a random patch if volumes are larger than box_size
        x, y, z = raw_vol.shape
        bx, by, bz = self.box_size
        
        if x >= bx and y >= by and z >= bz:
            # Extract random patch
            x_start = np.random.randint(0, x - bx + 1)
            y_start = np.random.randint(0, y - by + 1)
            z_start = np.random.randint(0, z - bz + 1)
            
            raw_patch = raw_vol[x_start:x_start+bx, y_start:y_start+by, z_start:z_start+bz]
            mask_patch = mask_vol[x_start:x_start+bx, y_start:y_start+by, z_start:z_start+bz]
        else:
            # If volumes are smaller than box_size, pad them
            raw_patch = np.zeros(self.box_size, dtype=raw_vol.dtype)
            mask_patch = np.zeros(self.box_size, dtype=mask_vol.dtype)
            
            # Copy as much as possible from the original volumes
            raw_patch[:min(bx, x), :min(by, y), :min(bz, z)] = raw_vol[:min(bx, x), :min(by, y), :min(bz, z)]
            mask_patch[:min(bx, x), :min(by, y), :min(bz, z)] = mask_vol[:min(bx, x), :min(by, y), :min(bz, z)]
        
        # Normalize raw data to [0, 1]
        raw_patch = (raw_patch - raw_patch.min()) / (raw_patch.max() - raw_patch.min() + 1e-8)
        
        # Ensure mask is binary (0 or 1)
        mask_patch = (mask_patch > 0).astype(np.float32)
        
        # Convert to PyTorch tensors and add channel dimension
        raw_tensor = torch.from_numpy(raw_patch).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)
        
        return raw_tensor, mask_tensor
    
    def _load_tiff_volume(self, tiff_dir):
        """Load a 3D volume from a stack of TIFF slices."""
        tiff_files = sorted([f for f in os.listdir(tiff_dir) if f.endswith('.tif')])
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {tiff_dir}")
        
        # Read first slice to get dimensions
        first_slice = tifffile.imread(os.path.join(tiff_dir, tiff_files[0]))
        height, width = first_slice.shape
        depth = len(tiff_files)
        
        # Create volume array
        dtype = first_slice.dtype
        volume = np.zeros((height, width, depth), dtype=dtype)
        
        # Load all slices
        for i, tiff_file in enumerate(tiff_files):
            slice_path = os.path.join(tiff_dir, tiff_file)
            slice_data = tifffile.imread(slice_path)
            volume[:, :, i] = slice_data
        
        return volume

def get_morphofeatures_texture_dataloader(root_dir, sample_ids=None, batch_size=4, shuffle=True, 
                                          num_workers=0, is_cytoplasm=False, box_size=(104, 104, 104),
                                          use_tiff=True, input_dir="raw", target_dir="mask",
                                          class_csv_path=None, filter_by_class=None):
    """
    Create a dataloader for texture data.
    
    Args:
        root_dir (str): Root directory containing the data
        sample_ids (list): List of sample directories (if None, will search in root_dir)
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for the dataloader
        is_cytoplasm (bool): Whether we're processing cytoplasm data
        box_size (tuple): Size of the 3D patches to extract
        use_tiff (bool): Whether to use TIFF files
        input_dir (str): Name of the directory containing input TIFF slices
        target_dir (str): Name of the directory containing target/mask TIFF slices
        class_csv_path (str): Path to CSV file containing sample information - not used directly here
        filter_by_class (int or list): Class IDs to include (None for all) - not used directly here
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for texture data
    """
    try:
        logger.info(f"Creating texture dataloader with {len(sample_ids) if sample_ids else 0} sample paths")
        
        # If no sample_ids provided, scan the root directory
        if not sample_ids:
            logger.warning("No sample_ids provided, scanning directories...")
            sample_ids = []
            if os.path.exists(root_dir):
                for dirname in os.listdir(root_dir):
                    path = os.path.join(root_dir, dirname)
                    if os.path.isdir(path):
                        raw_dir = os.path.join(path, input_dir)
                        mask_dir = os.path.join(path, target_dir)
                        if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                            sample_ids.append(path)
                            logger.info(f"Found valid sample directory: {path}")
            
            if not sample_ids:
                logger.error(f"No valid sample directories found in {root_dir}")
                raise ValueError(f"No valid sample directories found")
        
        # Verify that sample_ids exist and contain the required directories
        valid_samples = []
        for path in sample_ids:
            raw_dir = os.path.join(path, input_dir)
            mask_dir = os.path.join(path, target_dir)
            if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                valid_samples.append(path)
                logger.info(f"Verified sample directory: {path}")
            else:
                logger.warning(f"Directory missing required subdirectories: {path}")
        
        if not valid_samples:
            logger.error(f"None of the provided sample paths contain required subdirectories")
            raise ValueError(f"No valid sample directories found")
        
        logger.info(f"Creating dataset with {len(valid_samples)} valid samples")
        dataset = TiffVolumeDataset(
            sample_ids=valid_samples,
            is_cytoplasm=is_cytoplasm,
            box_size=box_size,
            input_dir=input_dir,
            target_dir=target_dir
        )
        
        logger.info(f"Creating dataloader with dataset of size {len(dataset)}")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return dataloader
    except Exception as e:
        logger.error(f"Error creating texture dataloader: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MorphoFeatures texture adapter")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--is_cytoplasm", action="store_true", help="Use cytoplasm box size (144³) instead of nucleus (104³)")
    args = parser.parse_args()
    
    # Get adapted dataloader
    dataloader = get_morphofeatures_texture_dataloader(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        batch_size=args.batch_size,
        is_cytoplasm=args.is_cytoplasm,
        debug=True
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Test the first batch
    for batch in dataloader:
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: (dict)")
                for k, v in value.items():
                    print(f"    {k}: {v}")
        
        # Verify the shapes are correct for MorphoFeatures
        if 'volume' in batch:
            volume_shape = batch['volume'].shape
            print(f"\nVerifying volume shape for MorphoFeatures texture model:")
            print(f"  Volume shape: {volume_shape}")
            box_size = 144 if args.is_cytoplasm else 104
            if volume_shape[1] == 1 and volume_shape[2] == box_size and volume_shape[3] == box_size and volume_shape[4] == box_size:
                print(f"\n✅ Texture dataloader has correct dimensions for {'cytoplasm' if args.is_cytoplasm else 'nucleus'} box!")
            else:
                print(f"\n❌ Texture dataloader does NOT have the correct dimensions.")
                print(f"  Expected: [B, 1, {box_size}, {box_size}, {box_size}]")
                print(f"  Got: {volume_shape}")
        
        # Only test the first batch
        break 