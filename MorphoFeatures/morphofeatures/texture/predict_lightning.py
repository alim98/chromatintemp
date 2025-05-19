import argparse
import os
import numpy as np
import torch
import z5py
import yaml
import pytorch_lightning as pl
from texture_lightning import TextureNet
from cell_loader import CellLoaders
import warnings


def predict(model, loader, path_to_save):
    """Generate and save embeddings for all cells
    
    Args:
        model: Trained Lightning model
        loader: Data loader with predict=True
        path_to_save: Path to save embeddings
    """
    pred_loader = loader.get_predict_loaders()
    labels = pred_loader.dataset.indices
    encoded = []
    batch = 0
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for samples in pred_loader:
            print(f"Processing batch {batch}")
            if loader.config.get('texture_contrastive', False):
                prediction = model(samples[0].to(device), just_encode=True).cpu().numpy()
                if np.any(np.isnan(prediction)):
                    warnings.warn("NaN detected in predictions")
                encoded.append(np.nanmean(prediction, axis=0))
            else:
                prediction = model(samples.to(device), just_encode=True).cpu().numpy()
                encoded.extend(list(prediction))
            batch += 1
            
    encoded = np.array(encoded)
    np.savetxt(path_to_save, np.c_[labels, encoded])
    print(f"Saved embeddings to {path_to_save}")


def predict_patches(model, loader, path_to_save):
    """Generate and save patch embeddings for textural patches
    
    Args:
        model: Trained Lightning model
        loader: Data loader with texture_contrastive=True
        path_to_save: Path to save patch embeddings
    """
    assert loader.config.get('texture_contrastive')
    pred_loader = loader.get_predict_loaders()
    bs = pred_loader.batch_size
    positions = pred_loader.dataset.positions
    
    # Create output file
    f = z5py.File(path_to_save)
    ds = f.create_dataset('preds', shape=(positions.shape[0], 80), dtype='float64',
                          compression='gzip')
    
    model.eval()
    device = next(model.parameters()).device
    
    batch = 0
    with torch.no_grad():
        for samples in pred_loader:
            print(f"Processing batch {batch}")
            prediction = model(samples.to(device), just_encode=True).cpu().numpy()
            if np.any(np.isnan(prediction)):
                warnings.warn("NaN detected in predictions")
            ds[batch * bs : batch * bs + prediction.shape[0]] = prediction
            batch += 1
            
    # Save cell IDs
    ids = pred_loader.dataset.positions[:, 0].astype('int64')
    ds = f.create_dataset('ids', data=ids, dtype='int64', compression='gzip')
    print(f"Saved patch embeddings to {path_to_save}")


def aggregate_patches(z5_path):
    """Aggregate patch embeddings by cell ID
    
    Args:
        z5_path: Path to z5 file with patch embeddings
    """
    path_to_save = os.path.dirname(z5_path) + '/avg_encoded_patches_aggr.np'
    f = z5py.File(z5_path)
    ids = f['ids'][:]
    labels = np.unique(ids)
    
    aggr_feat = np.zeros((labels.shape[0], f['preds'].shape[1]))
    for i, idx in enumerate(labels):
        patch_ids = np.where(ids == idx)[0]
        # they should be sequential
        assert np.all(patch_ids == np.arange(patch_ids[0], patch_ids[-1] + 1))
        aggr_feat[i] = np.mean(f['preds'][slice(patch_ids[0], patch_ids[-1] + 1)], axis=0)
        
    np.savetxt(path_to_save, np.c_[labels, aggr_feat])
    print(f"Saved aggregated embeddings to {path_to_save}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate cell embeddings using trained model')
    parser.add_argument('checkpoint_path', type=str,
                        help='Path to Lightning checkpoint (.ckpt file)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to test config file (if different from training)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save embeddings (defaults to checkpoint directory)')
    parser.add_argument('--save_patches', action='store_true',
                        help='For texture encoder, whether to save each patch')
    parser.add_argument('--aggregate_patches', action='store_true',
                        help='Aggregate patch features for each cell')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use for inference')
    args = parser.parse_args()

    # Set GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model from checkpoint
    model = TextureNet.load_from_checkpoint(args.checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Determine config file
    if args.config is None:
        # Try to find config in the checkpoint directory
        config_dir = os.path.dirname(args.checkpoint_path)
        if args.save_patches:
            config_path = os.path.join(config_dir, 'test_config_patches.yml')
        else:
            config_path = os.path.join(config_dir, 'test_config.yml')
            
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}. Please specify with --config")
    else:
        config_path = args.config
    
    # Path for embeddings
    encoded_path = os.path.join(args.output_dir, 'avg_encoded.np')
    if args.save_patches:
        encoded_path = encoded_path[:-3] + '_patches.z5'
    
    # Generate embeddings if they don't exist
    if not os.path.exists(encoded_path):
        cell_loader = CellLoaders(config_path)
        
        if args.save_patches:
            predict_patches(model, cell_loader, encoded_path)
        else:
            predict(model, cell_loader, encoded_path)
        
        print(f"Saved embeddings to {encoded_path}")
    
    # Aggregate patch features if requested
    if args.save_patches and args.aggregate_patches:
        aggregate_patches(encoded_path) 