import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from functools import partial
import math
from utils.volume_augmentations import random_flip, random_rotate90, elastic_deformation

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.highres_image_dataloader import get_highres_image_dataloader


def extract_cubes(volume, cube_size=32, min_foreground_percent=0.5, mask=None, max_cubes=32):
    """
    Extract cubes from a high-resolution volume, discarding cubes with insufficient foreground.
    
    Args:
        volume (torch.Tensor): Input volume of shape [Z, 1, H, W]
        cube_size (int): Size of the cubes to extract
        min_foreground_percent (float): Minimum percentage of foreground required (0.0-1.0)
        mask (torch.Tensor, optional): Binary mask to apply (1=foreground, 0=background)
        max_cubes (int, optional): Maximum number of cubes to extract (default 32)
        
    Returns:
        list: List of extracted cubes as torch tensors, each of shape [1, cube_size, cube_size, cube_size]
    """
    z_dim, c_dim, h_dim, w_dim = volume.shape
    print(f"Volume shape: {volume.shape}, cube_size: {cube_size}")
    
    # Validate dimensions
    min_spatial_dim = min(h_dim, w_dim)  # Only consider H,W for initial sizing
    if cube_size > min_spatial_dim:
        print(f"Warning: cube_size ({cube_size}) is larger than smallest spatial dimension ({min_spatial_dim})")
        # Adjust cube size to largest power of 2 that fits but is no larger than 1/2 the smallest dimension
        new_size = 32  # Start with minimum size
        target_max = min_spatial_dim // 2  # Don't use more than half the smallest dimension
        while new_size * 2 <= target_max and new_size * 2 <= 64:  # Cap at 64 to avoid memory issues
            new_size *= 2
        print(f"Adjusting cube_size to {new_size}")
        cube_size = new_size
    
    # Create a binary foreground mask if not provided
    if mask is None:
        # Lower threshold for more lenient foreground detection
        foreground_threshold = 0.05  # Reduced from 0.1
        mask = (volume > foreground_threshold).float()
        print(f"Created mask with threshold {foreground_threshold}, foreground ratio: {mask.mean().item():.3f}")
    
    # Calculate number of cubes that can fit in each dimension with 50% overlap
    z_steps = max(1, (z_dim - cube_size) // (cube_size // 2) + 1)
    h_steps = max(1, (h_dim - cube_size) // (cube_size // 2) + 1)
    w_steps = max(1, (w_dim - cube_size) // (cube_size // 2) + 1)
    
    print(f"Steps: z={z_steps}, h={h_steps}, w={w_steps}")
    print(f"Using cube_size={cube_size}, min_foreground={min_foreground_percent}")
    
    cubes = []
    total_checked = 0
    total_size_mismatch = 0
    total_foreground_reject = 0
    
    # Calculate stride based on how many cubes we want
    total_possible = z_steps * h_steps * w_steps
    if max_cubes is not None and total_possible > max_cubes:
        # Adjust stride to get closer to max_cubes
        stride_factor = math.ceil(math.sqrt(total_possible / max_cubes))
        z_stride = max(1, z_steps // stride_factor)
        h_stride = max(1, h_steps // stride_factor)
        w_stride = max(1, w_steps // stride_factor)
        print(f"Adjusted strides to z={z_stride}, h={h_stride}, w={w_stride} to target {max_cubes} cubes")
    else:
        z_stride = h_stride = w_stride = 1
    
    # Iterate through all possible cube positions with adjusted stride
    for z_idx in range(0, z_steps, z_stride):
        z_start = min(z_dim - cube_size, z_idx * (cube_size // 2))
        z_end = z_start + cube_size
        
        for h_idx in range(0, h_steps, h_stride):
            h_start = min(h_dim - cube_size, h_idx * (cube_size // 2))
            h_end = h_start + cube_size
            
            for w_idx in range(0, w_steps, w_stride):
                w_start = min(w_dim - cube_size, w_idx * (cube_size // 2))
                w_end = w_start + cube_size
                
                total_checked += 1
                
                # Extract the cube
                cube = volume[z_start:z_end, :, h_start:h_end, w_start:w_end]
                cube_mask = mask[z_start:z_end, :, h_start:h_end, w_start:w_end]
                
                # Skip if cube dimensions don't match (edge cases)
                if cube.shape[0] != cube_size or cube.shape[2] != cube_size or cube.shape[3] != cube_size:
                    if total_size_mismatch == 0:  # Print details for first mismatch only
                        print(f"Size mismatch: got {cube.shape}, expected [{cube_size}, 1, {cube_size}, {cube_size}]")
                    total_size_mismatch += 1
                    continue
                
                # Check if cube has sufficient foreground
                foreground_ratio = cube_mask.sum() / (cube_size**3)
                if foreground_ratio >= min_foreground_percent:
                    # Reshape to [1, Z, H, W] format for 3D convolutions
                    cube = cube.permute(1, 0, 2, 3)
                    cubes.append(cube)
                else:
                    total_foreground_reject += 1
                    
                # Break if we've reached the maximum number of cubes
                if max_cubes is not None and len(cubes) >= max_cubes:
                    print(f"Reached max cubes ({max_cubes})")
                    return cubes
    
    print(f"Extraction stats: checked={total_checked}, size_mismatch={total_size_mismatch}, "
          f"foreground_reject={total_foreground_reject}, accepted={len(cubes)}")
    
    # Random shuffle the cubes
    random.shuffle(cubes)
    
    # Limit to max_cubes if specified
    if max_cubes is not None and len(cubes) > max_cubes:
        cubes = cubes[:max_cubes]
    
    return cubes


def augment_cube(cube, p_rotate=0.7, p_flip=0.5, p_elastic=0.3):
    """
    Apply 3D augmentations to a cube: rotations, flips, and elastic deformations.
    
    Args:
        cube (torch.Tensor): Input cube of shape [1, Z, H, W]
        p_rotate (float): Probability of applying rotations
        p_flip (float): Probability of applying flips
        p_elastic (float): Probability of applying elastic deformations
        
    Returns:
        torch.Tensor: Augmented cube of the same shape
    """
    # Convert to numpy for easier manipulation
    cube_np = cube.squeeze(0).numpy()  # [Z, H, W]
    
    # 1. Random 90-degree rotations
    if random.random() < p_rotate:
        k = random.choice([1, 2, 3])  # Number of 90-degree rotations
        axes = random.choice([(0, 1), (0, 2), (1, 2)])  # Rotation axes
        cube_np = np.rot90(cube_np, k=k, axes=axes)
    
    # 2. Random flips
    if random.random() < p_flip:
        axis = random.randint(0, 2)  # Axis to flip (0=Z, 1=Y, 2=X)
        cube_np = np.flip(cube_np, axis=axis)
    
    # 3. Simple elastic-like deformation (intensity jitter)
    if random.random() < p_elastic:
        # Add random noise proportional to local intensity
        noise_level = 0.1  # Adjust based on your data
        noise = np.random.normal(0, noise_level, cube_np.shape) * (cube_np + 0.1)
        cube_np = np.clip(cube_np + noise, 0, 1)  # Keep in [0, 1] range
    
    # Make a contiguous copy to handle negative strides from flips/rotations
    cube_np = np.ascontiguousarray(cube_np)
    
    # Convert back to tensor
    return torch.tensor(cube_np).unsqueeze(0)  # [1, Z, H, W]


def create_contrastive_pairs(cubes, num_pairs=32):
    """
    Create contrastive pairs from a list of cubes from the same cell.
    
    Args:
        cubes (list): List of cube tensors from the same cell
        num_pairs (int): Number of contrastive pairs to create
        
    Returns:
        list: List of pairs, each containing two augmented views of a cube
    """
    if len(cubes) < 2:
        # If we have only one cube, create pairs using heavy augmentation
        pairs = []
        for _ in range(min(num_pairs, 1)):  # Limit to 1 pair if we have only 1 cube
            original = cubes[0]
            view1 = augment_cube(original, p_rotate=0.9, p_flip=0.7, p_elastic=0.5)
            view2 = augment_cube(original, p_rotate=0.9, p_flip=0.7, p_elastic=0.5)
            pairs.append((view1, view2))
        return pairs
    
    # Create pairs using different cubes
    pairs = []
    for _ in range(min(num_pairs, len(cubes) * (len(cubes) - 1) // 2)):
        # Sample two different cubes
        idx1, idx2 = random.sample(range(len(cubes)), 2)
        cube1, cube2 = cubes[idx1], cubes[idx2]
        
        # Apply augmentations
        view1 = augment_cube(cube1)
        view2 = augment_cube(cube2)
        
        pairs.append((view1, view2))
    
    return pairs


class HighResTextureDataset(Dataset):
    """
    Dataset for high-resolution texture feature extraction.
    Extracts 32³ cubes, generates contrastive pairs, and prepares them for the texture encoder.
    """
    def __init__(self, 
                 root_dir,
                 is_cytoplasm=False,
                 cube_size=32,  # Changed from 8 to match loader default
                 pairs_per_sample=32,
                 min_foreground_percent=0.1,  # Reduced from 0.5 to be more lenient
                 class_csv_path=None,
                 sample_ids=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 debug=False):
        """
        Args:
            root_dir (str): Root directory containing the samples
            is_cytoplasm (bool): If True, use cytoplasm masks; if False, use nucleus masks
            cube_size (int): Size of the cubes to extract (default 32³)
            pairs_per_sample (int): Number of contrastive pairs to create per sample
            min_foreground_percent (float): Minimum percentage of foreground required
            class_csv_path (str): Path to CSV file with class information
            sample_ids (list): List of sample IDs to include
            filter_by_class (int or list): Class ID(s) to include
            ignore_unclassified (bool): Whether to ignore unclassified samples
            debug (bool): Whether to print debug information
        """
        self.root_dir = root_dir
        self.is_cytoplasm = is_cytoplasm
        self.cube_size = cube_size
        self.pairs_per_sample = pairs_per_sample
        self.min_foreground_percent = min_foreground_percent
        self.debug = debug
        
        # Get the highres image dataloader to load raw volumes
        # Use a minimal window size to get the full volume
        self.highres_loader = get_highres_image_dataloader(
            root_dir=root_dir,
            batch_size=1,  # Process one sample at a time
            shuffle=False,  # No need to shuffle at this stage
            num_workers=0,  # Avoid multiprocessing issues
            class_csv_path=class_csv_path,
            sample_ids=sample_ids,
            filter_by_class=filter_by_class,
            ignore_unclassified=ignore_unclassified,
            z_window_size=cube_size * 4,  # Large enough window to extract multiple cubes
            debug=debug
        )
        
        self.dataset = self.highres_loader.dataset
        self.contrastive_pairs = []
        
        # Pre-extract contrastive pairs from each sample
        self._extract_all_pairs()
    
    def _extract_all_pairs(self):
        """Extract contrastive pairs from all samples in the dataset."""
        if self.debug:
            print(f"Extracting contrastive pairs from {len(self.dataset)} samples...")
            print(f"Parameters: cube_size={self.cube_size}, min_foreground={self.min_foreground_percent}")
        
        total_samples = 0
        total_cubes = 0
        total_pairs = 0
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            volume = sample['image']  # [Z, 1, H, W]
            sample_id = sample['metadata']['sample_id']
            
            if self.debug:
                print(f"\nProcessing sample {sample_id} ({idx+1}/{len(self.dataset)})")
                print(f"Volume shape: {volume.shape}")
            
            # TODO: Add proper masking for nucleus/cytoplasm when available
            # For now, we'll use a threshold-based mask
            mask = (volume > 0.05).float()  # Lowered threshold from 0.1
            
            # Extract cubes from this sample
            cubes = extract_cubes(
                volume=volume,
                cube_size=self.cube_size,
                min_foreground_percent=self.min_foreground_percent,
                mask=mask
            )
            
            if self.debug:
                print(f"Sample {sample_id}: Extracted {len(cubes)} valid cubes")
            
            total_samples += 1
            total_cubes += len(cubes)
            
            # Skip samples with no valid cubes
            if not cubes:
                continue
            
            # Create contrastive pairs
            pairs = create_contrastive_pairs(cubes, num_pairs=self.pairs_per_sample)
            total_pairs += len(pairs)
            
            # Store the pairs with their metadata
            for pair in pairs:
                self.contrastive_pairs.append({
                    'views': pair,
                    'sample_id': sample_id,
                    'class_id': sample['label'],
                    'is_cytoplasm': self.is_cytoplasm
                })
        
        if self.debug:
            print(f"\nDataset statistics:")
            print(f"Total samples processed: {total_samples}")
            print(f"Total cubes extracted: {total_cubes}")
            print(f"Total contrastive pairs: {total_pairs}")
            print(f"Average cubes per sample: {total_cubes/total_samples:.1f}")
            print(f"Average pairs per sample: {total_pairs/total_samples:.1f}")
    
    def __len__(self):
        return len(self.contrastive_pairs)
    
    def __getitem__(self, idx):
        pair_data = self.contrastive_pairs[idx]
        view1, view2 = pair_data['views']
        
        return {
            'view1': view1,  # [1, Z, H, W]
            'view2': view2,  # [1, Z, H, W]
            'label': pair_data['class_id'],
            'metadata': {
                'sample_id': pair_data['sample_id'],
                'is_cytoplasm': pair_data['is_cytoplasm']
            }
        }


def collate_contrastive_batch(batch):
    """
    Collate function for contrastive batches.
    
    Args:
        batch (list): List of dictionaries, each with 'view1', 'view2' tensors
        
    Returns:
        dict: Batch with stacked views and metadata
    """
    view1_list = [item['view1'] for item in batch]
    view2_list = [item['view2'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Collect metadata
    sample_ids = [item['metadata']['sample_id'] for item in batch]
    is_cytoplasm = [item['metadata']['is_cytoplasm'] for item in batch]
    
    return {
        'view1': torch.stack(view1_list, dim=0),  # [B, 1, Z, H, W]
        'view2': torch.stack(view2_list, dim=0),  # [B, 1, Z, H, W]
        'label': torch.tensor(labels),
        'metadata': {
            'sample_id': sample_ids,
            'is_cytoplasm': is_cytoplasm
        }
    }


def adapt_batch_for_morphofeatures_highres(batch):
    """
    Adapt a batch for MorphoFeatures high-resolution texture model.
    
    Args:
        batch (dict): A batch from the HighResTextureDataset
        
    Returns:
        list: List of tensors in the format expected by the MorphoFeatures model:
              [view1, view2] for contrastive learning
    """
    # Extract views
    view1 = batch['view1']  # [B, 1, Z, H, W]
    view2 = batch['view2']  # [B, 1, Z, H, W]
    
    # Return views for the MorphoFeatures texture model
    return [view1, view2]


def adapted_highres_collate_fn(original_collate_fn, batch):
    """
    Adapter for the highres collate function that can be pickled.
    
    Args:
        original_collate_fn: Original collate function
        batch: Batch to process
        
    Returns:
        Adapted batch
    """
    original_batch = original_collate_fn(batch)
    return adapt_batch_for_morphofeatures_highres(original_batch)


def adapt_highres_dataloader_for_morphofeatures(dataloader):
    """
    Adapt the highres dataloader for MorphoFeatures texture model.
    
    Args:
        dataloader: DataLoader instance from get_highres_texture_dataloader
        
    Returns:
        DataLoader: New DataLoader with adapted collate function
    """
    # Create a new dataloader with adapted collate function
    original_collate_fn = dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else collate_contrastive_batch
    
    # Use a functools.partial to create a new function with the original collate_fn bound
    from functools import partial
    adapted_fn = partial(adapted_highres_collate_fn, original_collate_fn)
    
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


def get_morphofeatures_highres_texture_dataloader(
    root_dir,
    batch_size=32,  # As specified in the paper
    shuffle=True,
    num_workers=0,  # Default to 0 to avoid multiprocessing issues
    is_cytoplasm=False,
    cube_size=32,  # Default to 32 as in paper, override with smaller size if needed
    pairs_per_sample=32,
    min_foreground_percent=0.5,
    class_csv_path=None,
    sample_ids=None,
    filter_by_class=None,
    ignore_unclassified=True,
    pin_memory=False,
    debug=False,
    box_size=None  # Add box_size parameter
):
    """
    Get a highres texture dataloader already adapted for MorphoFeatures texture model.
    
    This creates a dataloader that provides 32³ voxel cubes from high-resolution
    volumes (20×20×25nm) as specified in the paper, with appropriate contrastive pairs
    for the NT-Xent loss.
    
    Args:
        root_dir (str): Root directory containing the samples
        batch_size (int): Batch size (paper uses 32)
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes (use 0 to avoid multiprocessing issues)
        is_cytoplasm (bool): If True, use cytoplasm masks; if False, use nucleus masks
        cube_size (int): Size of the cubes to extract (paper uses 32³)
        pairs_per_sample (int): Number of contrastive pairs to create per sample
        min_foreground_percent (float): Minimum percentage of foreground required
        class_csv_path (str): Path to CSV file with class information
        sample_ids (list): List of sample IDs to include
        filter_by_class (int or list): Class ID(s) to include
        ignore_unclassified (bool): Whether to ignore unclassified samples
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        debug (bool): Whether to print debug information
        box_size (list): Optional box size from config, used to determine appropriate cube size
        
    Returns:
        DataLoader: Adapted dataloader for MorphoFeatures high-resolution texture model
    """
    # If box_size is provided, use a cube size that's a power of 2 and fits within the smallest dimension
    if box_size is not None:
        min_dim = min(box_size)  # Get smallest dimension
        # Find largest power of 2 that's <= min_dim and >= 32
        cube_size = 32  # Start with minimum size from paper
        while cube_size * 2 <= min_dim and cube_size * 2 <= 128:  # Cap at 128 to avoid memory issues
            cube_size *= 2
        if debug:
            print(f"Adjusted cube_size to {cube_size} based on box_size {box_size}")
    
    # Create the contrastive dataset
    dataset = HighResTextureDataset(
        root_dir=root_dir,
        is_cytoplasm=is_cytoplasm,
        cube_size=cube_size,
        pairs_per_sample=pairs_per_sample,
        min_foreground_percent=min_foreground_percent,
        class_csv_path=class_csv_path,
        sample_ids=sample_ids,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        debug=debug
    )
    
    # Create the dataloader with custom collate function
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch,
        pin_memory=pin_memory
    )
    
    # Adapt the dataloader for MorphoFeatures
    return adapt_highres_dataloader_for_morphofeatures(dataloader)


if __name__ == "__main__":
    # Example usage
    dataloader = get_morphofeatures_highres_texture_dataloader(
        root_dir="data",
        batch_size=4,
        class_csv_path="chromatin_classes_and_samples.csv",
        debug=True
    )
    
    # Process one batch
    for batch in dataloader:
        print(f"Batch shapes: view1={batch[0].shape}, view2={batch[1].shape}")
        break 