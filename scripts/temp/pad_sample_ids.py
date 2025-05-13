import pandas as pd
import os

def pad_sample_ids(input_csv, output_csv):
    """
    Create a new CSV file with zero-padded sample IDs (4 digits).
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Print initial info
    print(f"Original sample IDs (first 5): {df['sample_id'].head().tolist()}")
    
    # Convert sample IDs to 4-digit zero-padded strings
    df['sample_id'] = df['sample_id'].apply(lambda x: f"{int(x):04d}")
    
    # Print converted info
    print(f"Converted sample IDs (first 5): {df['sample_id'].head().tolist()}")
    
    # Save to new file
    print(f"Saving padded sample IDs to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Done!")
    
    # Print unique sample IDs for verification
    unique_ids = sorted(df['sample_id'].unique())
    print(f"\nTotal unique sample IDs: {len(unique_ids)}")
    print("First 10 unique sample IDs:", unique_ids[:10])
    
    return unique_ids  # Return the list of unique IDs for verification

if __name__ == "__main__":
    # Define paths
    input_csv = os.path.join('data', 'chromatin_classes_and_samples2.csv')
    output_csv = os.path.join('data', 'chromatin_classes_and_samples2_padded.csv')
    
    # Create new CSV with padded sample IDs
    unique_ids = pad_sample_ids(input_csv, output_csv) 