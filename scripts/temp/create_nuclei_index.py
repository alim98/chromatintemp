import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import sys

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def create_nuclei_index(data_dir, output_csv, class_csv_path=None):
    """
    Creates a CSV index file for all nuclei samples in the dataset.
    
    Args:
        data_dir (str): Path to the nuclei dataset directory
        output_csv (str): Path where the output CSV file will be saved
        class_csv_path (str, optional): Path to CSV file with class information
    
    Returns:
        DataFrame: The created index as a pandas DataFrame
    """
    print(f"Creating nuclei index from directory: {data_dir}")
    
    # Load class information if available
    class_info = {}
    if class_csv_path and os.path.exists(class_csv_path):
        print(f"Loading class information from: {class_csv_path}")
        class_df = pd.read_csv(class_csv_path)
        for _, row in class_df.iterrows():
            # Skip unclassified samples
            if row['class_name'] == 'Unclassified' or row['class_id'] == 19:
                continue
                
            sample_id = str(row['sample_id']).zfill(4)  # Ensure 4-digit format
            class_info[sample_id] = {
                'class_id': row['class_id'],
                'class_name': row['class_name']
            }
        print(f"Loaded class information for {len(class_info)} samples (excluding unclassified)")
    
    # Find all sample directories (using 4-digit format)
    all_dirs = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))])
    
    # Filter to only include directories with 4-digit names
    sample_dirs = []
    for d in all_dirs:
        # Handle both cases: already has 4 digits or needs zero-padding
        if d.isdigit():
            sample_dirs.append(d.zfill(4))
        else:
            sample_dirs.append(d)
    
    print(f"Found {len(sample_dirs)} potential sample directories")
    
    # Prepare data for DataFrame
    data = []
    
    for sample_id in sample_dirs:
        # When looking for the directory, we need to handle both zero-padded and non-zero-padded
        # because the directory might exist in either format
        raw_dir = os.path.join(data_dir, sample_id, 'raw')
        if not os.path.exists(raw_dir) and sample_id.isdigit():
            # Try with non-zero-padded version if 4-digit version doesn't exist
            raw_dir = os.path.join(data_dir, sample_id.lstrip('0'), 'raw')
            if not os.path.exists(raw_dir):
                print(f"Warning: Raw directory not found for sample {sample_id}")
                continue
        
        mask_dir = os.path.join(data_dir, sample_id, 'mask')
        if not os.path.exists(mask_dir) and sample_id.isdigit():
            # Try with non-zero-padded version if 4-digit version doesn't exist
            mask_dir = os.path.join(data_dir, sample_id.lstrip('0'), 'mask')
            if not os.path.exists(mask_dir):
                print(f"Warning: Mask directory not found for sample {sample_id}")
                continue
        
        # Skip if not in class info (this will include unclassified samples)
        if sample_id not in class_info:
            # Skip this sample as it's either unclassified or not in our class list
            continue
        
        # Get raw image files
        raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
        
        if not raw_files:
            print(f"Warning: No raw files found for sample {sample_id}")
            continue
        
        # Print some debug info for the first few samples
        if len(data) < 5:
            print(f"Sample {sample_id}: Found {len(raw_files)} raw files")
            print(f"  First file: {os.path.basename(raw_files[0])}")
            print(f"  Last file: {os.path.basename(raw_files[-1])}")
        
        # Get dimensions from first image to describe the volume
        try:
            first_img = np.array(Image.open(raw_files[0]))
            height, width = first_img.shape
            depth = len(raw_files)
            
            # Get class information 
            class_id = class_info[sample_id]['class_id']
            class_name = class_info[sample_id]['class_name']
            
            # Add to data
            data.append({
                'sample_id': sample_id,
                'raw_dir': raw_dir,
                'mask_dir': mask_dir,
                'depth': depth,
                'height': height,
                'width': width,
                'num_files': len(raw_files),
                'class_id': class_id,
                'class_name': class_name,
                'first_file': raw_files[0]
            })
            
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Created nuclei index with {len(df)} valid samples")
    print(f"Saved to: {output_csv}")
    
    # Print class distribution
    class_counts = df['class_name'].value_counts()
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create a CSV index of nuclei samples')
    
    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT,
                        help='Path to the nuclei dataset directory')
    parser.add_argument('--output_csv', type=str, default=os.path.join(config.ANALYSIS_OUTPUT_DIR, 'nuclei_index.csv'),
                        help='Path where the output CSV file will be saved')
    parser.add_argument('--class_csv', type=str, default=config.CLASS_CSV_PATH,
                        help='Path to CSV file with class information')
    
    args = parser.parse_args()
    
    create_nuclei_index(args.data_dir, args.output_csv, args.class_csv)

if __name__ == "__main__":
    main() 