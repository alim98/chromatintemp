import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

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

def get_morphofeatures_texture_dataloader(
    root_dir,
    batch_size=8,
    shuffle=True,
    num_workers=0,  # Default to 0 to avoid multiprocessing issues
    class_csv_path=None,
    sample_ids=None,
    filter_by_class=None,
    ignore_unclassified=True,
    box_size=(104, 104, 104),  # Size of cropped box (nucleus default)
    is_cytoplasm=False,        # If True, use 144³ box instead of 104³
    z_window_size=80,
    pin_memory=False,
    debug=False
):
    """
    Get a lowres texture dataloader already adapted for MorphoFeatures texture model.
    
    This creates a dataloader that provides 3D volumes with effective voxel size of 
    80x80x100nm, and crops boxes of either 104³ (nucleus) or 144³ (cytoplasm)
    as specified in the paper.
    
    Args:
        root_dir (str): Root directory containing the samples
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes (defaults to 0 to avoid multiprocessing issues)
        class_csv_path (str): Path to CSV file with class information
        sample_ids (list): List of sample IDs to include
        filter_by_class (int or list): Class ID(s) to include
        ignore_unclassified (bool): Whether to ignore unclassified samples
        box_size (tuple): Size of cropped box in voxels (H, W, D)
        is_cytoplasm (bool): If True, use 144³ box instead of 104³
        z_window_size (int): Number of Z slices to include
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        debug (bool): Whether to print debug information
        
    Returns:
        DataLoader: Adapted dataloader for MorphoFeatures texture model
    """
    # Adjust box size if cytoplasm
    if is_cytoplasm:
        box_size = (144, 144, 144)
    
    # Create transform function using a proper named function instead of lambda
    transform = make_transform_function((box_size[0], box_size[1]))
    
    # Create the original lowres dataloader
    dataloader = get_lowres_image_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # Use 0 to avoid multiprocessing issues
        transform=transform,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        target_size=(box_size[0], box_size[1]),
        sample_percent=100,
        z_window_size=box_size[2],  # Use box_size[2] for z dimension
        pin_memory=pin_memory,
        debug=debug
    )
    
    # Adapt the dataloader for MorphoFeatures
    return adapt_lowres_dataloader_for_morphofeatures(dataloader)

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