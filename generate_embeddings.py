#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm

# Add MorphoFeatures to the path
sys.path.append(os.path.abspath("MorphoFeatures"))

from MorphoFeatures.morphofeatures.nn.embeddings import MorphoFeaturesExtractor
from dataloader.morphofeatures_adapter import get_morphofeatures_mesh_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate 480-D MorphoFeatures embeddings from trained models")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for data")
    parser.add_argument("--class_csv", type=str, required=True, help="CSV file with class information")
    parser.add_argument("--output_dir", type=str, default="embeddings/full", help="Output directory for embeddings")
    parser.add_argument("--cyto_shape_model", type=str, help="Path to cytoplasm shape model")
    parser.add_argument("--cyto_coarse_texture_model", type=str, help="Path to cytoplasm coarse texture model")
    parser.add_argument("--cyto_fine_texture_model", type=str, help="Path to cytoplasm fine texture model")
    parser.add_argument("--nucleus_shape_model", type=str, help="Path to nucleus shape model")
    parser.add_argument("--nucleus_coarse_texture_model", type=str, help="Path to nucleus coarse texture model")
    parser.add_argument("--nucleus_fine_texture_model", type=str, help="Path to nucleus fine texture model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    return parser.parse_args()

def get_data_generator(args):
    """Create a data generator that yields batches of data"""
    # Cytoplasm mesh dataloader
    cyto_dataloader = get_morphofeatures_mesh_dataloader(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        filter_by_class=None,
        ignore_unclassified=True,
        num_points=1024,  # As per paper spec
        debug=False
    )
    
    # TODO: Add more dataloaders for texture volumes
    # This would depend on your adapters for coarse/fine texture loading
    
    # Generate batches
    for cyto_batch in cyto_dataloader:
        batch_data = {
            'cyto_points': cyto_batch['points'],
            'cyto_features': cyto_batch['features'],
            # Add other data tensors here once texture loaders are implemented
        }
        
        # Get sample IDs from batch
        if 'metadata' in cyto_batch and 'sample_id' in cyto_batch['metadata']:
            sample_ids = cyto_batch['metadata']['sample_id']
        else:
            # Generate sample IDs if not available in batch
            sample_ids = [f"sample_{i}" for i in range(len(cyto_batch['points']))]
        
        yield batch_data, sample_ids

def main():
    args = parse_args()
    
    # Create the MorphoFeatures extractor with model paths
    extractor = MorphoFeaturesExtractor(
        cyto_shape_model_path=args.cyto_shape_model,
        cyto_coarse_texture_model_path=args.cyto_coarse_texture_model,
        cyto_fine_texture_model_path=args.cyto_fine_texture_model,
        nucleus_shape_model_path=args.nucleus_shape_model,
        nucleus_coarse_texture_model_path=args.nucleus_coarse_texture_model,
        nucleus_fine_texture_model_path=args.nucleus_fine_texture_model,
        device=args.device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data generator
    data_generator = get_data_generator(args)
    
    # Process batches
    extractor.batch_process(data_generator, args.output_dir)
    
    logger.info(f"Embeddings saved to {args.output_dir}")

if __name__ == "__main__":
    main() 