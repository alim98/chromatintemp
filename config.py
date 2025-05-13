import os

# Project paths
DATA_ROOT = os.path.join("data", "nuclei_sample_1a_v1")
CLASS_CSV_PATH = os.path.join("data", "chromatin_classes_and_samples.csv")
RESULTS_DIR = "results"
ANALYSIS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis_output")
VISUALIZATION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualizations")

# Data sampling configuration
DATA_SUBSAMPLE_RATIO = 0.1  # Use 10% of the entire dataset for training/validation

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
