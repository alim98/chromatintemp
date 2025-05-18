import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from MorphoFeatures.morphofeatures.shape.network import DeepGCN
from MorphoFeatures.morphofeatures.nn.texture_encoder import TextureEncoder


class MorphoFeaturesExtractor:
    """
    Inference-only extractor that generates 480-D MorphoFeatures vectors by running
    6 different models and concatenating their outputs.
    
    The six models are:
    1. Cytoplasm shape (80-D)
    2. Cytoplasm coarse texture (80-D)
    3. Cytoplasm fine texture (80-D)
    4. Nucleus shape (80-D)
    5. Nucleus coarse texture (80-D)
    6. Nucleus fine texture (80-D)
    """
    def __init__(
        self,
        cyto_shape_model_path: Optional[str] = None,
        cyto_coarse_texture_model_path: Optional[str] = None, 
        cyto_fine_texture_model_path: Optional[str] = None,
        nucleus_shape_model_path: Optional[str] = None,
        nucleus_coarse_texture_model_path: Optional[str] = None,
        nucleus_fine_texture_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        # Set device
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.models = {}
        
        # Load shape models if provided
        if cyto_shape_model_path is not None and os.path.exists(cyto_shape_model_path):
            self.models['cyto_shape'] = self._load_shape_model(cyto_shape_model_path)
        
        if nucleus_shape_model_path is not None and os.path.exists(nucleus_shape_model_path):
            self.models['nucleus_shape'] = self._load_shape_model(nucleus_shape_model_path)
            
        # Load texture models if provided
        if cyto_coarse_texture_model_path is not None and os.path.exists(cyto_coarse_texture_model_path):
            self.models['cyto_coarse_texture'] = self._load_texture_model(cyto_coarse_texture_model_path)
            
        if cyto_fine_texture_model_path is not None and os.path.exists(cyto_fine_texture_model_path):
            self.models['cyto_fine_texture'] = self._load_texture_model(cyto_fine_texture_model_path)
            
        if nucleus_coarse_texture_model_path is not None and os.path.exists(nucleus_coarse_texture_model_path):
            self.models['nucleus_coarse_texture'] = self._load_texture_model(nucleus_coarse_texture_model_path)
            
        if nucleus_fine_texture_model_path is not None and os.path.exists(nucleus_fine_texture_model_path):
            self.models['nucleus_fine_texture'] = self._load_texture_model(nucleus_fine_texture_model_path)
            
    def _load_shape_model(self, model_path: str) -> nn.Module:
        """Load a DeepGCN shape model from checkpoint"""
        print(f"Loading shape model from {model_path}")
        # Create model
        model = DeepGCN(out_channels=80, projection_head=False)
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        # Set to eval mode and move to device
        model.eval()
        model.to(self.device)
        return model
    
    def _load_texture_model(self, model_path: str) -> nn.Module:
        """Load a TextureEncoder model from checkpoint"""
        print(f"Loading texture model from {model_path}")
        # Create model
        model = TextureEncoder(out_channels=80)
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        # Set to eval mode and move to device
        model.eval()
        model.to(self.device)
        return model
    
    def extract_shape_features(
        self, 
        model_key: str, 
        points: torch.Tensor, 
        features: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Extract shape features from a point cloud
        
        Args:
            model_key: Key for the model to use ('cyto_shape' or 'nucleus_shape')
            points: Point cloud tensor [B, 3, N]
            features: Optional features tensor [B, 6, N]
            
        Returns:
            Feature array [B, 80]
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.models[model_key]
        
        # Process batch
        with torch.no_grad():
            # Make sure points are on the right device
            points = points.to(self.device)
            if features is not None:
                features = features.to(self.device)
            
            # Forward pass
            _, embeddings = model(points, features)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
        
        return embeddings_np
    
    def extract_texture_features(
        self, 
        model_key: str, 
        volume: torch.Tensor
    ) -> np.ndarray:
        """
        Extract texture features from a 3D volume
        
        Args:
            model_key: Key for the model to use (e.g., 'cyto_coarse_texture')
            volume: Volume tensor [B, 1, D, H, W]
            
        Returns:
            Feature array [B, 80]
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.models[model_key]
        
        # Process batch
        with torch.no_grad():
            # Make sure volume is on the right device
            volume = volume.to(self.device)
            
            # Forward pass
            _, embeddings = model(volume)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
        
        return embeddings_np
    
    def extract_all_features(
        self,
        cyto_points: Optional[torch.Tensor] = None,
        cyto_features: Optional[torch.Tensor] = None,
        cyto_coarse_volume: Optional[torch.Tensor] = None,
        cyto_fine_volume: Optional[torch.Tensor] = None,
        nucleus_points: Optional[torch.Tensor] = None,
        nucleus_features: Optional[torch.Tensor] = None,
        nucleus_coarse_volume: Optional[torch.Tensor] = None,
        nucleus_fine_volume: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Extract all features; tolerate missing cytoplasm data.
        """
        branch_embeddings = {
            'cyto_shape': None,
            'cyto_coarse': None,
            'cyto_fine': None,
            'nuc_shape': None,
            'nuc_coarse': None,
            'nuc_fine': None,
        }

        # Nucleus first since cytoplasm may be absent
        if nucleus_points is not None and 'nucleus_shape' in self.models:
            branch_embeddings['nuc_shape'] = self.extract_shape_features('nucleus_shape', nucleus_points, nucleus_features)
        if nucleus_coarse_volume is not None and 'nucleus_coarse_texture' in self.models:
            branch_embeddings['nuc_coarse'] = self.extract_texture_features('nucleus_coarse_texture', nucleus_coarse_volume)
        if nucleus_fine_volume is not None and 'nucleus_fine_texture' in self.models:
            branch_embeddings['nuc_fine'] = self.extract_texture_features('nucleus_fine_texture', nucleus_fine_volume)

        # Cytoplasm (optional)
        if cyto_points is not None and 'cyto_shape' in self.models:
            branch_embeddings['cyto_shape'] = self.extract_shape_features('cyto_shape', cyto_points, cyto_features)
        if cyto_coarse_volume is not None and 'cyto_coarse_texture' in self.models:
            branch_embeddings['cyto_coarse'] = self.extract_texture_features('cyto_coarse_texture', cyto_coarse_volume)
        if cyto_fine_volume is not None and 'cyto_fine_texture' in self.models:
            branch_embeddings['cyto_fine'] = self.extract_texture_features('cyto_fine_texture', cyto_fine_volume)

        # Determine batch size from first available embedding
        for emb in branch_embeddings.values():
            if emb is not None:
                batch_size = emb.shape[0]
                break
        else:
            batch_size = 1  # Fallback

        # Replace missing branches with zeros
        for key, emb in branch_embeddings.items():
            if emb is None:
                branch_embeddings[key] = np.zeros((batch_size, 80), dtype=np.float32)

        # Concatenate in correct order
        ordered = [
            branch_embeddings['cyto_shape'],
            branch_embeddings['cyto_coarse'],
            branch_embeddings['cyto_fine'],
            branch_embeddings['nuc_shape'],
            branch_embeddings['nuc_coarse'],
            branch_embeddings['nuc_fine'],
        ]

        return np.concatenate(ordered, axis=1)
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str, sample_id: str = None):
        """
        Save embeddings to disk
        
        Args:
            embeddings: Embeddings array
            output_path: Output directory
            sample_id: Optional sample ID for filename
        """
        os.makedirs(output_path, exist_ok=True)
        filename = f"{sample_id}.npy" if sample_id else "embeddings.npy"
        np.save(os.path.join(output_path, filename), embeddings)
    
    def batch_process(self, data_generator, output_path: str, max_samples: int = None):
        """
        Process a batch of data and save embeddings
        
        Args:
            data_generator: Generator that yields data and sample IDs
            output_path: Output directory
            max_samples: Maximum number of samples to process
        """
        os.makedirs(output_path, exist_ok=True)
        
        sample_count = 0
        for batch_data, sample_ids in data_generator:
            # Extract features
            embeddings = self.extract_all_features(**batch_data)
            
            # Save embeddings for each sample
            for i, sample_id in enumerate(sample_ids):
                if i < embeddings.shape[0]:
                    self.save_embeddings(
                        embeddings[i:i+1], 
                        output_path, 
                        sample_id
                    )
                    sample_count += 1
                    print(f"Processed sample {sample_id} ({sample_count} total)")
            
            # Check if we reached the maximum
            if max_samples is not None and sample_count >= max_samples:
                break
        
        print(f"Finished processing {sample_count} samples")


# For testing
if __name__ == "__main__":
    # Create a dummy extractor
    extractor = MorphoFeaturesExtractor()
    
    # Create dummy data
    batch_size = 2
    cyto_points = torch.randn(batch_size, 3, 1024)
    cyto_features = torch.randn(batch_size, 6, 1024)
    cyto_coarse_volume = torch.randn(batch_size, 1, 144, 144, 144)
    cyto_fine_volume = torch.randn(batch_size, 1, 32, 32, 32)
    
    nucleus_points = torch.randn(batch_size, 3, 1024)
    nucleus_features = torch.randn(batch_size, 6, 1024)
    nucleus_coarse_volume = torch.randn(batch_size, 1, 144, 144, 144)
    nucleus_fine_volume = torch.randn(batch_size, 1, 32, 32, 32)
    
    # Since no models are loaded, this will return zeros
    features = extractor.extract_all_features(
        cyto_points=cyto_points,
        cyto_features=cyto_features,
        cyto_coarse_volume=cyto_coarse_volume,
        cyto_fine_volume=cyto_fine_volume,
        nucleus_points=nucleus_points,
        nucleus_features=nucleus_features,
        nucleus_coarse_volume=nucleus_coarse_volume,
        nucleus_fine_volume=nucleus_fine_volume
    )
    
    print(f"Generated features shape: {features.shape}")  # Should be [2, 480] 