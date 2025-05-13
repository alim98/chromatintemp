import pandas as pd
import os
import shutil
from tqdm import tqdm

def copy_sample_folders(csv_path, source_dir, dest_dir):
    """
    Copy sample folders based on sample IDs from CSV file.
    
    Args:
        csv_path (str): Path to CSV file containing sample IDs
        source_dir (str): Source directory containing sample folders
        dest_dir (str): Destination directory for copied folders
    """
    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get unique sample IDs (they are already zero-padded in the new CSV)
    sample_ids = sorted(df['sample_id'].unique())
    print(f"Found {len(sample_ids)} unique sample IDs")
    print("First 5 sample IDs:", sample_ids[:5])
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    
    # Copy folders
    print("\nCopying folders...")
    copied = 0
    errors = 0
    error_list = []
    
    for folder_name in tqdm(sample_ids):
        src_path = os.path.join(source_dir, folder_name)
        dst_path = os.path.join(dest_dir, folder_name)
        
        try:
            if os.path.exists(src_path):
                # If destination already exists, remove it
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                
                # Copy the folder
                shutil.copytree(src_path, dst_path)
                copied += 1
            else:
                error_msg = f"Source folder not found: {src_path}"
                print(f"Warning: {error_msg}")
                error_list.append((folder_name, error_msg))
                errors += 1
        except Exception as e:
            error_msg = f"Error copying folder {folder_name}: {str(e)}"
            print(f"Error: {error_msg}")
            error_list.append((folder_name, error_msg))
            errors += 1
    
    # Print summary
    print("\nCopy operation completed!")
    print(f"Successfully copied: {copied} folders")
    print(f"Errors encountered: {errors} folders")
    
    if errors > 0:
        print("\nError details:")
        for folder_name, error_msg in error_list:
            print(f"  {folder_name}: {error_msg}")

if __name__ == "__main__":
    # Define paths
    csv_path = os.path.join('data', 'chromatin_classes_and_samples2_padded.csv')
    source_dir = os.path.join('data', 'nuclei_sample_1a_v2')
    dest_dir = os.path.join('data', 'nuclei_sample_1a_v1')
    
    # Copy the folders
    copy_sample_folders(csv_path, source_dir, dest_dir) 