import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataloaders
from dataloader.lowres_image_dataloader import get_lowres_image_dataloader, LowResImageDataset
from dataloader.highres_image_dataloader import get_highres_image_dataloader
from dataloader.lowres_texture_adapter import get_morphofeatures_texture_dataloader as get_lowres_texture_dataloader
from dataloader.highres_texture_adapter import get_morphofeatures_highres_texture_dataloader as get_highres_texture_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Verify downsampling in lowres vs highres data")
    parser.add_argument("--sample_id", type=str, default=None, help="Specific sample ID to check")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv_path", type=str, default="chromatin_classes_and_samples.csv", 
                        help="CSV file with class and sample information")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    print("Loading lowres and highres dataloaders...")
    
    # Get raw image dataloaders first to check the original data
    print("Loading raw image dataloaders...")
    lowres_image_dataloader = get_lowres_image_dataloader(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sample_ids=[args.sample_id] if args.sample_id else None
    )
    
    highres_image_dataloader = get_highres_image_dataloader(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv_path,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sample_ids=[args.sample_id] if args.sample_id else None
    )
    
    # Get texture dataloaders (these may have additional processing)
    print("Loading texture dataloaders...")
    try:
        lowres_texture_dataloader = get_lowres_texture_dataloader(
            root_dir=args.root_dir,
            class_csv_path=args.class_csv_path,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sample_ids=[args.sample_id] if args.sample_id else None
        )
        
        highres_texture_dataloader = get_highres_texture_dataloader(
            root_dir=args.root_dir,
            class_csv_path=args.class_csv_path,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sample_ids=[args.sample_id] if args.sample_id else None
        )
        has_texture_dataloaders = True
    except Exception as e:
        print(f"Error loading texture dataloaders: {e}")
        print("Continuing with only image dataloaders")
        has_texture_dataloaders = False
    
    # Get a single sample from raw image dataloaders
    lowres_image_batch = next(iter(lowres_image_dataloader))
    highres_image_batch = next(iter(highres_image_dataloader))
    
    # Debug print to see the batch structure
    print("\nBatch structure:")
    print(f"Lowres batch type: {type(lowres_image_batch)}")
    if isinstance(lowres_image_batch, dict):
        print(f"Lowres batch keys: {list(lowres_image_batch.keys())}")
    elif isinstance(lowres_image_batch, (list, tuple)):
        print(f"Lowres batch length: {len(lowres_image_batch)}")
        for i, item in enumerate(lowres_image_batch):
            print(f"Item {i} type: {type(item)}")
            if hasattr(item, 'shape'):
                print(f"Item {i} shape: {item.shape}")
    
    # Extract raw image data based on batch structure
    if isinstance(lowres_image_batch, dict) and 'image' in lowres_image_batch:
        lowres_volume = lowres_image_batch['image'].cpu().numpy()
    elif isinstance(lowres_image_batch, (list, tuple)) and len(lowres_image_batch) > 0:
        if isinstance(lowres_image_batch[0], dict) and 'image' in lowres_image_batch[0]:
            lowres_volume = lowres_image_batch[0]['image'].cpu().numpy()
        else:
            lowres_volume = lowres_image_batch[0].cpu().numpy()
    else:
        raise ValueError("Unexpected lowres batch format")
    
    if isinstance(highres_image_batch, dict) and 'image' in highres_image_batch:
        highres_volume = highres_image_batch['image'].cpu().numpy()
    elif isinstance(highres_image_batch, (list, tuple)) and len(highres_image_batch) > 0:
        if isinstance(highres_image_batch[0], dict) and 'image' in highres_image_batch[0]:
            highres_volume = highres_image_batch[0]['image'].cpu().numpy()
        else:
            highres_volume = highres_image_batch[0].cpu().numpy()
    else:
        raise ValueError("Unexpected highres batch format")
    
    # Ensure volumes are in the expected format [C, D, H, W]
    if lowres_volume.ndim == 3:  # [D, H, W]
        lowres_volume = np.expand_dims(lowres_volume, axis=0)
    elif lowres_volume.ndim == 4 and lowres_volume.shape[0] == 1:  # [1, D, H, W]
        pass  # Already in the right format
    else:
        print(f"Warning: Unexpected lowres volume shape: {lowres_volume.shape}")
    
    if highres_volume.ndim == 3:  # [D, H, W]
        highres_volume = np.expand_dims(highres_volume, axis=0)
    elif highres_volume.ndim == 4 and highres_volume.shape[0] == 1:  # [1, D, H, W]
        pass  # Already in the right format
    else:
        print(f"Warning: Unexpected highres volume shape: {highres_volume.shape}")
    
    # Get sample ID
    sample_id = args.sample_id if args.sample_id else "random"
    
    # Print basic information
    print(f"\nSample ID: {sample_id}")
    print(f"Raw lowres volume shape: {lowres_volume.shape}")
    print(f"Raw highres volume shape: {highres_volume.shape}")
    
    # Calculate statistics for raw data
    print("\nRaw pixel value statistics:")
    print(f"Lowres - Min: {lowres_volume.min():.4f}, Max: {lowres_volume.max():.4f}, Mean: {lowres_volume.mean():.4f}")
    print(f"Highres - Min: {highres_volume.min():.4f}, Max: {highres_volume.max():.4f}, Mean: {highres_volume.mean():.4f}")
    
    # Take middle slices of raw data
    print("\nExtracting middle slices for visualization...")
    
    # For lowres volume with shape (1, 80, 1, 64, 64)
    if lowres_volume.ndim == 5:
        # Shape is [B, Z, C, H, W]
        z_mid = lowres_volume.shape[1] // 2
        lowres_mid_slice = lowres_volume[0, z_mid, 0]  # [H, W]
        print(f"Lowres middle slice shape: {lowres_mid_slice.shape}")
    elif lowres_volume.ndim == 4:
        # Shape is [C, Z, H, W]
        z_mid = lowres_volume.shape[1] // 2
        lowres_mid_slice = lowres_volume[0, z_mid]  # [H, W]
    elif lowres_volume.ndim == 3:
        # Shape is [Z, H, W]
        z_mid = lowres_volume.shape[0] // 2
        lowres_mid_slice = lowres_volume[z_mid]  # [H, W]
    else:
        print(f"Error: Lowres volume has unexpected dimensions: {lowres_volume.ndim}")
        return
    
    # For highres volume with shape (1, 80, 1, 224, 224)
    if highres_volume.ndim == 5:
        # Shape is [B, Z, C, H, W]
        z_mid = highres_volume.shape[1] // 2
        highres_mid_slice = highres_volume[0, z_mid, 0]  # [H, W]
        print(f"Highres middle slice shape: {highres_mid_slice.shape}")
    elif highres_volume.ndim == 4:
        # Shape is [C, Z, H, W]
        z_mid = highres_volume.shape[1] // 2
        highres_mid_slice = highres_volume[0, z_mid]  # [H, W]
    elif highres_volume.ndim == 3:
        # Shape is [Z, H, W]
        z_mid = highres_volume.shape[0] // 2
        highres_mid_slice = highres_volume[z_mid]  # [H, W]
    else:
        print(f"Error: Highres volume has unexpected dimensions: {highres_volume.ndim}")
        return
    
    # Create figure to compare raw data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show lowres slice
    axes[0].imshow(lowres_mid_slice, cmap='gray')
    axes[0].set_title(f"Lowres (80×80×100nm)\nShape: {lowres_mid_slice.shape}")
    
    # Show highres slice
    axes[1].imshow(highres_mid_slice, cmap='gray')
    axes[1].set_title(f"Highres (20×20×25nm)\nShape: {highres_mid_slice.shape}")
    
    # Create a downsampled version of highres to compare
    # Assuming 4x downsampling factor
    downsampling_factor = 4
    
    print("\nCreating downsampled version of highres volume...")
    
    # For highres volume with shape (1, 80, 1, 224, 224)
    if highres_volume.ndim == 5:
        # Extract the relevant parts: [B, Z, C, H, W] -> [Z, H, W]
        highres_for_downsampling = highres_volume[0, :, 0, :, :]  # [Z, H, W]
    elif highres_volume.ndim == 4:
        highres_for_downsampling = highres_volume[0]  # [Z, H, W] or [C, Z, H, W]
        if highres_for_downsampling.shape[0] == 1:  # [C, Z, H, W]
            highres_for_downsampling = highres_for_downsampling[0]  # [Z, H, W]
    else:
        highres_for_downsampling = highres_volume  # Assume it's already [Z, H, W]
    
    print(f"Highres for downsampling shape: {highres_for_downsampling.shape}")
    
    # Ensure dimensions are divisible by downsampling factor
    d, h, w = highres_for_downsampling.shape
    
    # Adjust dimensions to be divisible by downsampling factor
    d_adj = (d // downsampling_factor) * downsampling_factor
    h_adj = (h // downsampling_factor) * downsampling_factor
    w_adj = (w // downsampling_factor) * downsampling_factor
    
    if d != d_adj or h != h_adj or w != w_adj:
        print(f"Adjusting dimensions from {(d, h, w)} to {(d_adj, h_adj, w_adj)} to be divisible by {downsampling_factor}")
        highres_for_downsampling = highres_for_downsampling[:d_adj, :h_adj, :w_adj]
    
    try:
        # Reshape to prepare for downsampling
        reshaped = highres_for_downsampling.reshape(
            d_adj // downsampling_factor, downsampling_factor,
            h_adj // downsampling_factor, downsampling_factor,
            w_adj // downsampling_factor, downsampling_factor
        )
        
        # Average over the downsampling windows
        downsampled = reshaped.mean(axis=(1, 3, 5))
        
        print(f"Downsampled shape: {downsampled.shape}")
        
        # Get middle slice of downsampled volume
        downsampled_mid_slice = downsampled[downsampled.shape[0]//2]
        
        # Compare with original lowres
        print("\nDownsampling comparison:")
        
        # Extract lowres data for comparison
        if lowres_volume.ndim == 5:
            lowres_for_comparison = lowres_volume[0, :, 0, :, :]  # [Z, H, W]
        elif lowres_volume.ndim == 4:
            lowres_for_comparison = lowres_volume[0]  # [Z, H, W] or [C, Z, H, W]
            if lowres_for_comparison.shape[0] == 1:  # [C, Z, H, W]
                lowres_for_comparison = lowres_for_comparison[0]  # [Z, H, W]
        else:
            lowres_for_comparison = lowres_volume  # Assume it's already [Z, H, W]
        
        print(f"Lowres for comparison shape: {lowres_for_comparison.shape}")
        
        # Calculate similarity metrics if shapes allow
        if lowres_for_comparison.shape == downsampled.shape:
            # Calculate mean squared error
            mse = np.mean((lowres_for_comparison - downsampled) ** 2)
            print(f"Mean Squared Error between lowres and downsampled highres: {mse:.6f}")
            
            # Calculate structural similarity (if available)
            try:
                from skimage.metrics import structural_similarity as ssim
                # Calculate SSIM for each z-slice and average
                ssim_values = []
                for z in range(lowres_for_comparison.shape[0]):
                    ssim_val = ssim(lowres_for_comparison[z], downsampled[z])
                    ssim_values.append(ssim_val)
                avg_ssim = np.mean(ssim_values)
                print(f"Average Structural Similarity (SSIM): {avg_ssim:.6f}")
            except ImportError:
                print("skimage not available for SSIM calculation")
        else:
            print(f"Shape mismatch: Cannot compare lowres {lowres_for_comparison.shape} with downsampled {downsampled.shape}")
    except Exception as e:
        print(f"Error during downsampling: {e}")
        downsampled_mid_slice = None
        downsampled = None
    
    # Show downsampled highres
    if downsampled_mid_slice is not None:
        axes[2].imshow(downsampled_mid_slice, cmap='gray')
        axes[2].set_title(f"Highres downsampled by {downsampling_factor}x\nShape: {downsampled_mid_slice.shape}")
    else:
        axes[2].text(0.5, 0.5, "Downsampling failed", ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("Downsampling Error")
    
    # Add overall title
    plt.suptitle(f"Downsampling Verification for Sample {sample_id}", fontsize=16)
    
    # Save figure
    output_path = f"visualizations/downsampling_verification_{sample_id}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")
    
    # Additional analysis: histogram comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.hist(lowres_volume.flatten(), bins=50, alpha=0.7)
    plt.title("Lowres Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 2)
    plt.hist(highres_volume.flatten(), bins=50, alpha=0.7)
    plt.title("Highres Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    try:
        plt.subplot(1, 3, 3)
        if downsampled is not None:
            plt.hist(downsampled.flatten(), bins=50, alpha=0.7)
            plt.title("Downsampled Highres Histogram")
        else:
            plt.text(0.5, 0.5, "Downsampling failed", ha='center', va='center')
            plt.title("Downsampled Highres Histogram")
    except NameError:
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, "Downsampling failed", ha='center', va='center')
        plt.title("Downsampled Highres Histogram")
    
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # Save histogram figure
    hist_output_path = f"visualizations/downsampling_histograms_{sample_id}.png"
    plt.tight_layout()
    plt.savefig(hist_output_path)
    print(f"Histogram comparison saved to {hist_output_path}")
    
    # If texture dataloaders are available, also analyze those
    if has_texture_dataloaders:
        print("\nAnalyzing texture dataloader outputs...")
        
        # Get samples from texture dataloaders
        lowres_texture_batch = next(iter(lowres_texture_dataloader))
        highres_texture_batch = next(iter(highres_texture_dataloader))
        
        # Extract texture data
        if isinstance(lowres_texture_batch, (list, tuple)):
            lowres_texture = lowres_texture_batch[0][0].cpu().numpy()  # [C, D, H, W]
        else:
            lowres_texture = lowres_texture_batch[0].cpu().numpy()  # [C, D, H, W]
            
        if isinstance(highres_texture_batch, (list, tuple)):
            highres_texture = highres_texture_batch[0][0].cpu().numpy()  # [C, D, H, W]
        else:
            highres_texture = highres_texture_batch[0].cpu().numpy()  # [C, D, H, W]
        
        print(f"Texture lowres shape: {lowres_texture.shape}")
        print(f"Texture highres shape: {highres_texture.shape}")
        
        # Create figure for texture data
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show middle slices
        lowres_texture_mid = lowres_texture[0, lowres_texture.shape[1]//2] if lowres_texture.ndim > 3 else lowres_texture[lowres_texture.shape[0]//2]
        highres_texture_mid = highres_texture[0, highres_texture.shape[1]//2] if highres_texture.ndim > 3 else highres_texture[highres_texture.shape[0]//2]
        
        axes[0].imshow(lowres_texture_mid, cmap='gray')
        axes[0].set_title(f"Lowres Texture\nShape: {lowres_texture.shape}")
        
        axes[1].imshow(highres_texture_mid, cmap='gray')
        axes[1].set_title(f"Highres Texture\nShape: {highres_texture.shape}")
        
        # Save texture figure
        texture_output_path = f"visualizations/texture_comparison_{sample_id}.png"
        plt.tight_layout()
        plt.savefig(texture_output_path)
        print(f"Texture comparison saved to {texture_output_path}")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main() 