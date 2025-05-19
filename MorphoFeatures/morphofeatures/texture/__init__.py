# MorphoFeatures texture module

# Make key classes available at the module level
from .texture_lightning import TextureNet, make_loaders, make_predict_loader
from .cell_loader import CellLoaders, get_transforms
from .cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset

__all__ = [
    'TextureNet',
    'CellLoaders',
    'RawAEContrCellDataset',
    'TextPatchContrCellDataset',
    'get_transforms',
    'make_loaders',
    'make_predict_loader'
]

# Import key functions from cell_loader.py
try:
    from .cell_loader import collate_contrastive, CellLoaders
except ImportError:
    # Provide minimal implementation if the original is not available
    import torch
    def collate_contrastive(batch):
        """Collate function for contrastive learning"""
        inputs = torch.cat([i[0] for i in batch])
        targets = torch.cat([i[1] for i in batch])
        if len(batch[0]) == 3:
            targets2 = torch.cat([i[2] for i in batch])
            targets = [targets, targets2]
        return inputs, targets
    
    class CellLoaders:
        """Minimal implementation of CellLoaders"""
        def __init__(self, *args, **kwargs):
            pass
        
        def get_train_loaders(self):
            return None, None
        
        def get_predict_loaders(self):
            return None

# Import key functions from train.py
try:
    from .train import compile_criterion, set_up_training
except ImportError:
    # Provide minimal implementation if the original is not available
    def compile_criterion(config):
        """Minimal implementation of compile_criterion"""
        import torch.nn as nn
        return nn.CrossEntropyLoss()
    
    def set_up_training(project_directory, config):
        """Minimal implementation of set_up_training"""
        from torch.optim import Adam
        try:
            from inferno.trainers.basic import Trainer
            trainer = Trainer(model=None)
            trainer.build_optimizer(Adam, lr=0.001)
            trainer.build_criterion(compile_criterion(config))
            return trainer
        except ImportError:
            return None 