import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import monai
import yaml
from torch.utils.data import DataLoader

# Try relative import first, then fallback to absolute import
try:
    from .cell_loader import CellLoaders
except ImportError:
    try:
        # Fallback to package import
        from morphofeatures.texture.cell_loader import CellLoaders
    except ImportError:
        # Direct import as a last resort
        from cell_loader import CellLoaders

try:
    from morphofeatures.nn.losses import NTXentLoss, MorphoFeaturesLoss
except ImportError:
    # Fallback to direct import
    import sys
    import os.path as osp
    sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))
    from morphofeatures.nn.losses import NTXentLoss, MorphoFeaturesLoss

class TextureNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        
        # Get model configuration
        model_config = cfg.get('model_config', {})
        feature_dim = model_config.get('feature_dim', 80)  # Default feature dimension
        
        # backbone = 3-D UNet from MONAI (coarse texture)
        self.encoder = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=feature_dim,
            channels=(64, 128, 256),
            strides=(2, 2, 2),
            norm='batch'
        )
        
        # simple mirror decoder for the AE path
        self.decoder = monai.networks.nets.BasicUNet(
            spatial_dims=3, 
            in_channels=feature_dim, 
            out_channels=1
        )
        
        # Loss functions
        self.criterion_con = NTXentLoss(temperature=cfg.get('temperature', 0.1))
        self.criterion_rec = nn.MSELoss()
        self.lambda_rec = cfg.get('lambda_rec', 1.0)

    # ───────────────────────── Lightning API ──────────────────────────
    def forward(self, x, just_encode=False):
        """Forward pass through the model
        
        Args:
            x: Input volume (B, 1, D, H, W)
            just_encode: If True, only return the embeddings (for inference)
            
        Returns:
            If just_encode=True: embeddings (B, feature_dim)
            Otherwise: (embeddings, reconstructed volume)
        """
        # Encode and global average pool
        z = self.encoder(x).mean(dim=(-1, -2, -3))  # global-avg-pool to (B, feature_dim)
        
        if just_encode:  # inference mode used by predict.py
            return z
            
        # Reshape for decoder
        z_3d = z.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Expand to spatial dimensions expected by decoder
        z_3d = z_3d.expand(-1, -1, 1, 1, 1)
        # Decode to reconstruct
        recon = self.decoder(z_3d)
        
        return z, recon

    def training_step(self, batch, batch_idx):
        """Training step
        
        Args:
            batch: Tuple of ((aug1, aug2), (target1, target2))
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Unpack the batch
        (aug1, aug2), (target1, target2) = batch  # see CellLoader collate
        
        # Process augmented views
        z1, rec1 = self(aug1)
        z2, rec2 = self(aug2)
        
        # Compute contrastive loss between embeddings
        con_loss = self.criterion_con(z1, z2)
        
        # Compute reconstruction losses
        rec_loss1 = self.criterion_rec(rec1, target1)
        rec_loss2 = self.criterion_rec(rec2, target2)
        rec_loss = (rec_loss1 + rec_loss2) / 2
        
        # Compute combined loss
        loss = con_loss + self.lambda_rec * rec_loss
        
        # Log losses
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_con_loss", con_loss, prog_bar=False)
        self.log("train_rec_loss", rec_loss, prog_bar=False)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step
        
        Args:
            batch: Tuple of ((aug1, aug2), (target1, target2))
            batch_idx: Batch index
        """
        # Unpack the batch
        (aug1, aug2), (target1, target2) = batch
        
        # Process augmented views
        z1, rec1 = self(aug1)
        z2, rec2 = self(aug2)
        
        # Compute contrastive loss between embeddings
        con_loss = self.criterion_con(z1, z2)
        
        # Compute reconstruction losses
        rec_loss1 = self.criterion_rec(rec1, target1)
        rec_loss2 = self.criterion_rec(rec2, target2)
        rec_loss = (rec_loss1 + rec_loss2) / 2
        
        # Compute combined loss
        loss = con_loss + self.lambda_rec * rec_loss
        
        # Log losses
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_con_loss", con_loss, prog_bar=False)
        self.log("val_rec_loss", rec_loss, prog_bar=False)
        
        return loss

    def configure_optimizers(self):
        """Configure optimizers and LR schedulers
        
        Returns:
            Tuple of (optimizers, schedulers)
        """
        # Get optimizer configuration
        opt_config = self.hparams.get('optimizer_config', {})
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        # Create optimizer
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Create scheduler
        sched_config = self.hparams.get('scheduler_config', {})
        if sched_config.get('type', '') == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, 
                T_max=self.trainer.max_epochs
            )
            return [opt], [{"scheduler": sched, "interval": "epoch"}]
        elif sched_config.get('type', '') == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
            return [opt], [{"scheduler": sched, "interval": "epoch", "monitor": "val_loss"}]
        else:
            return opt


# ───────────────────────── Data side ──────────────────────────
def make_loaders(cfg_path):
    """Create data loaders from configuration file
    
    Args:
        cfg_path: Path to the config file
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    loader_builder = CellLoaders(cfg_path)
    train_dl, val_dl = loader_builder.get_train_loaders()
    return train_dl, val_dl


def make_predict_loader(cfg_path):
    """Create prediction loader from configuration file
    
    Args:
        cfg_path: Path to the config file
        
    Returns:
        Prediction dataloader
    """
    loader_builder = CellLoaders(cfg_path)
    pred_dl = loader_builder.get_predict_loaders()
    return pred_dl


if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train texture encoder with PyTorch Lightning')
    parser.add_argument('--train_config', type=str, default='train_config.yml',
                       help='Path to training configuration file')
    parser.add_argument('--data_config', type=str, default='data_config.yml',
                       help='Path to data configuration file')
    parser.add_argument('--output_dir', type=str, default='lightning_outputs',
                       help='Output directory for checkpoints and logs')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.train_config, 'r') as f:
        train_cfg = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Merge configurations
    cfg = {**train_cfg, **data_cfg}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    train_dl, val_dl = make_loaders(args.data_config)
    
    # Create model
    model = TextureNet(cfg)
    
    # Create Lightning trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator="gpu",
        devices=cfg.get("num_gpus", 1),
        max_epochs=cfg.get("num_epochs", 100),
        precision="16-mixed" if cfg.get("use_amp", False) else "32",
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss", 
                save_top_k=3,
                filename="{epoch}-{val_loss:.4f}"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ]
    )
    
    # Train model
    trainer.fit(model, train_dl, val_dl) 