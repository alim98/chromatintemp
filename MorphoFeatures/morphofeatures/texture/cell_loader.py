import os
import numpy as np
import pandas as pd
import z5py
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from monai.transforms import (
    Compose, SpatialPad, RandSpatialCrop, RandRotate90, Rand3DElastic,
    EnsureType, ScaleIntensityRange
)

from pybdv.metadata import get_data_path

# Try different import methods
try:
    # Try relative import first
    from .cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset 
except ImportError:
    try:
        # Try absolute import
        from morphofeatures.texture.cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset
    except ImportError:
        # Fallback to direct import
        from cell_dset import RawAEContrCellDataset, TextPatchContrCellDataset


def get_train_val_split(labels, split=0.2, r_seed=None):
    np.random.seed(seed=r_seed)
    np.random.shuffle(labels)
    spl = int(np.floor(len(labels)*split))
    return labels[spl:], labels[:spl]


def get_transforms(transform_config):
    """Build a MONAI transform pipeline from config"""
    transforms = []
    
    # Handle crop_pad_to_size (CropPad2Size replacement)
    if transform_config.get('crop_pad_to_size'):
        crop_pad_config = transform_config.get('crop_pad_to_size')
        size = crop_pad_config.get('size')
        mode = crop_pad_config.get('mode', 'constant')
        transforms.append(SpatialPad(spatial_size=size, mode=mode))
    
    # Handle random_crop (VolumeRandomCrop replacement)
    if transform_config.get('random_crop'):
        random_crop_config = transform_config.get('random_crop')
        size = random_crop_config.get('size')
        transforms.append(RandSpatialCrop(roi_size=size, random_center=True, random_size=False))
    
    # Handle cast (Cast replacement + ensure tensor)
    if transform_config.get('cast'):
        transforms.append(EnsureType(dtype=torch.float32, track_meta=False))
    
    # Handle normalize_range (NormalizeRange replacement)
    if transform_config.get('normalize_range'):
        normalize_config = transform_config.get('normalize_range')
        min_val = normalize_config.get('min_val', 0)
        max_val = normalize_config.get('max_val', 1)
        transforms.append(ScaleIntensityRange(a_min=min_val, a_max=max_val, b_min=0.0, b_max=1.0, clip=True))
    
    # Handle rotate90 (RandomRot903D replacement)
    if transform_config.get('rotate90'):
        transforms.append(RandRotate90(prob=0.5, spatial_axes=[0, 1, 2]))
    
    # Handle elastic_transform (ElasticTransform replacement)
    if transform_config.get('elastic_transform'):
        elastic_config = transform_config.get('elastic_transform')
        
        # Get parameters with appropriate defaults
        sigma_range = elastic_config.get('sigma_range', (5, 7))
        if not isinstance(sigma_range, tuple):
            sigma_range = (sigma_range, sigma_range)
            
        magnitude_range = elastic_config.get('magnitude_range', (50, 150))
        if not isinstance(magnitude_range, tuple):
            magnitude_range = (magnitude_range, magnitude_range)
        
        transforms.append(Rand3DElastic(
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=1.0,
            rotate_range=None,
            translate_range=None,
            scale_range=None,
            mode="bilinear",
            padding_mode="zeros"
        ))
    
    # MONAI returns tensors by default, so we don't need AsTorchBatch
    
    return Compose(transforms)


def collate_contrastive(batch):
    inputs = torch.cat([i[0] for i in batch])
    targets = torch.cat([i[1] for i in batch])
    if len(batch[0]) == 3:
        targets2 = torch.cat([i[2] for i in batch])
        targets = [targets, targets2]
    return inputs, targets


class CellLoaders(object):
    def __init__(self, configuration_file):
        # Replace yaml2dict with native yaml.safe_load
        with open(configuration_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        data_config = self.config.get('data_config')

        self.PATH = "/scratch/zinchenk/cell_match/data/platy_data"
        version = data_config.get("version")

        raw_data = os.path.join(self.PATH, "rawdata/sbem-6dpf-1-whole-raw.n5")
        cell_segm = os.path.join(self.PATH, version, "images/local",
                                 "sbem-6dpf-1-whole-segmented-cells.xml")
        nucl_segm = os.path.join(self.PATH, version, "images/local",
                                 "sbem-6dpf-1-whole-segmented-nuclei.xml")
        cell_to_nucl = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-cells/cells_to_nuclei.tsv")
        cell_default = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-cells/default.tsv")
        nucl_default = os.path.join(self.PATH, version, "tables",
                                    "sbem-6dpf-1-whole-segmented-nuclei/default.tsv")

        cell_file = z5py.File(get_data_path(cell_segm, True), 'r')
        nucl_file = z5py.File(get_data_path(nucl_segm, True), 'r')
        raw_file = z5py.File(raw_data, 'r')

        self.raw_vol = raw_file['setup0/timepoint0/s3']
        self.cell_vol = cell_file['setup0/timepoint0/s2']
        self.nuclei_vol = nucl_file['setup0/timepoint0/s0']

        self.nucl_dict = {int(k): int(v)
                          for k, v in np.loadtxt(cell_to_nucl, skiprows=1)
                          if v != 0}
        self.tables = [pd.read_csv(f, sep='\t') for f in [cell_default, nucl_default]]

        self.split = data_config.get('split', None)
        self.seed = data_config.get('seed', None)

        self.other_kwargs = self.config['other'] if 'other' in self.config else {}

        if self.config.get('contrastive', False):
            self.dset = RawAEContrCellDataset
        elif self.config.get('texture_contrastive', False):
            self.dset = TextPatchContrCellDataset
            raw_level = data_config.get("raw_level")
            self.raw_vol = raw_file['setup0/timepoint0/s{}'.format(raw_level)]
            self.other_kwargs['cell_hr_vol'] = cell_file['setup0/timepoint0/s{}'\
                                               .format(raw_level - 1)]

        self.transf = get_transforms(self.config.get('transforms')) \
                      if self.config.get('transforms') else None
        self.trans_sim = get_transforms(self.config.get('transforms_sim')) \
                         if self.config.get('transforms_sim') else None

    def get_train_loaders(self):
        labels = get_train_val_split(list(self.nucl_dict.keys()),
                                     split=self.split, r_seed=self.seed)
        cell_dsets = [self.dset(self.tables, self.nucl_dict,
                                self.cell_vol, self.nuclei_vol, self.raw_vol,
                                indices=i, transforms=self.transf,
                                transforms_sim=self.trans_sim,
                                **self.other_kwargs) for i in labels]

        train_loader = DataLoader(cell_dsets[0], collate_fn=collate_contrastive,
                                  **self.config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], collate_fn=collate_contrastive,
                                **self.config.get('val_loader_config'))
        return train_loader, val_loader

    def get_predict_loaders(self):
        pred_dataset = self.dset(self.tables, self.nucl_dict,
                                 self.cell_vol, self.nuclei_vol, self.raw_vol,
                                 transforms=self.transf, predict=True,
                                 **self.other_kwargs)
        pred_loader = DataLoader(pred_dataset, **self.config.get('pred_loader_config'))
        return pred_loader
