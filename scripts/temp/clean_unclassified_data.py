import pandas as pd
import os

def clean_unclassified_data(input_csv, output_csv=None):
    """
    Remove unclassified rows from the CSV file.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file. If None, will modify input file
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Print initial stats
    total_rows = len(df)
    print(f"Total rows before cleaning: {total_rows}")
    
    # Remove rows where class_name contains 'unclassified' (case insensitive)
    df_cleaned = df[~df['class_name'].str.lower().str.contains('unclassified', na=False)]
    
    # Calculate stats
    removed_rows = total_rows - len(df_cleaned)
    print(f"Rows removed: {removed_rows}")
    print(f"Rows remaining: {len(df_cleaned)}")
    
    # Save the cleaned data
    if output_csv is None:
        output_csv = input_csv
        
    # Create backup of original file if we're overwriting
    if output_csv == input_csv:
        backup_file = input_csv + '.backup'
        print(f"Creating backup of original file: {backup_file}")
        df.to_csv(backup_file, index=False)
    
    # Save cleaned data
    print(f"Saving cleaned data to: {output_csv}")
    df_cleaned.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    # Define input/output paths
    input_csv = os.path.join('data', 'chromatin_classes_and_samples2.csv')
    
    # Clean the data (will create backup and modify original file)
    clean_unclassified_data(input_csv) 