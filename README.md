## we dont have Cyto labels. 


Legend  
✓ = implemented & wired correctly  
△ = present but needs a small tweak to be 1-to-1 with the paper  
✗ = still missing / must be added if you want a perfect replica  

--------------------------------------------------------------------
1 DATA LAYER  (*paper § Pre-processing & Fig 1A left*)  
--------------------------------------------------------------------
| Pipeline element (paper)                       | our code                                  | Status |
| --------------------------------------------- | ------------------------------------------ | ------ |
| Point-cloud loader (XYZ + normals, B×1024×6)  | `mesh_dataloader.py` → `get_mesh_dataloader_v2`<br> + adapter in `dataloader/morphofeatures_adapter.py` | ✓ |
| Coarse-texture volume loader 144³ @ 80 nm      | `dataloader/lowres_texture_adapter.py` → `TiffVolumeDataset` (box = 104³/144³) | ✓ |
| Fine-texture patch loader 32³ @ 20 nm          | `dataloader/highres_texture_adapter.py` → `HighResTextureDataset` | ✓ |
| On-the-fly point-cloud aug. (rot/scale/ARAP)   | `shape/augmentations/arap.py`, `simple_transforms.py` **present** – plug-in via adapter still TODO | △ |
| On-the-fly volume aug. (rot 90°, flip, elastic) | `utils/volume_augmentations.py` + calls in both texture adapters | ✓ |

--------------------------------------------------------------------
2 SELF-SUPERVISED OBJECTIVES  (*paper Fig 1B,C*)  
--------------------------------------------------------------------
| Component                                       | our code                                                  | Status |
| ----------------------------------------------- | ---------------------------------------------------------- | ------ |
| NT-Xent contrastive loss (T = 0.07)             | `nn/losses.py → NTXentLoss`                                | ✓ |
| Auto-encoder reconstruction loss (MSE)          | handled inside `MorphoFeaturesLoss` for texture branches   | ✓ |
| λ-weighted total loss                           | `MorphoFeaturesLoss` implements \(L_{NT-Xent}+λ_{AE}·MSE+λ_{norm}\) | ✓ |
| λ values (shape = 0, texture ≈ 1e0, norm ≈ 1e-2)| defaults 0/1/1e-2 in `get_shape_loss` / `get_texture_loss` | ✓ |

--------------------------------------------------------------------
3 NETWORKS  (*paper § “Neural-network model”*)  
--------------------------------------------------------------------
| Encoder (paper)                               | our implementation                         | Out-dim | Status |
| --------------------------------------------- | ------------------------------------------- | ------- | ------ |
| DeepGCN (shape)                               | `shape/network/deepgcn.py` (3 GENConv)      | **80**  | △ ¹ |
| Coarse-/Fine-texture 3-block UNet encoder      | `nn/texture_encoder.py`                     | 80      | ✓ |
| Weight sharing (fine nucleus ← cyto)           | `TextureEncoder.transfer_weights` + `share_weights_from` in trainer | ✓ |

¹ The paper text (Suppl.) lists 64-D for shape; the most recent repo version switched to 80-D.  
If you must keep **exact** parity with the publication, set `out_channels=64` in `DeepGCN`.

--------------------------------------------------------------------
4 OPTIMISATION & SCHEDULING  (*paper Methods § Training*)  
--------------------------------------------------------------------
| Hyper-parameter (paper)                               | Shape | Texture | our default | Status |
| ----------------------------------------------------- | ----- | ------- | ------------ | ------ |
| Adam LR                                               | 2×10⁻⁴ | 1×10⁻⁴ | configurable; defaults injected if missing | ✓ |
| Weight-decay 4×10⁻⁴                                   | ✓     | ✓       | defaulted when not given                   | ✓ |
| Batch-size                                            | 96 clouds / 12-16 / 32 | user config | need to set in YAML | △ |
| Stop-on-plateau (patience ≈ 0.95 epochs)              | ReduceLROnPlateau(patience=5)            | ✓ |

--------------------------------------------------------------------
5 TRAINING ENTRY-POINTS  (*paper “analysis/train/…” scripts*)  
--------------------------------------------------------------------
| Branch (paper CLI)   | our runner                              | Writes |
| -------------------- | ---------------------------------------- | ------ |
| Shape                | `train_morphofeatures_models.py --shape_config …` | checkpoints & logs |
| Coarse / Fine tex.   | same file `--lowres_config` / `--highres_config`  | checkpoints & logs |

(single script instead of 3, but functionality identical) → ✓

--------------------------------------------------------------------
6 INFERENCE & MERGE  (*paper Fig 1A right*)  
--------------------------------------------------------------------
| Step                                                  | our code                                | Status |
| ---------------------------------------------------- | ---------------------------------------- | ------ |
| Forward six encoders; concat 480 D                   | `nn/embeddings.MorphoFeaturesExtractor`  | ✓ |
| Zero-padding when cytoplasm missing (nucleus-only)   | implemented; keeps 480-vector shape      | ✓ |
| Save `{id}.npy`                                      | `generate_embeddings.py`                 | ✓ |
| MorphoContextFeatures (avg. neighbours)              | `morphofeatures/context_features.py`     | ✓ |
