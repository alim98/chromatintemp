import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add MorphoFeatures to the path so we can import from it
sys.path.append(os.path.abspath("MorphoFeatures"))

from dataloader.mesh_dataloader import get_mesh_dataloader_v2
from dataloader.morphofeatures_adapter import adapt_mesh_dataloader_for_morphofeatures

def main():
    parser = argparse.ArgumentParser(description="Test mesh dataloader with DeepGCN")
    parser.add_argument("--root_dir", type=str, default="data", help="Root directory for data")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv", help="CSV with class info")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points per mesh")
    args = parser.parse_args()
    
    # Create a dataloader
    dataloader = get_mesh_dataloader_v2(
        root_dir=args.root_dir,
        class_csv_path=args.class_csv,
        batch_size=args.batch_size,
        return_type='pointcloud',
        num_points=args.num_points,
        debug=True
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Try to import DeepGCN directly
    try:
        # Try to import the model directly from the path
        sys.path.insert(0, os.path.abspath("."))
        from MorphoFeatures.morphofeatures.shape.network.deepgcn import DeepGCN
        print("Successfully imported DeepGCN from MorphoFeatures!")
        
        # Create the model
        model = DeepGCN(
            in_channels=6,      # 3D coordinates + normals
            channels=64,        # Hidden dimension
            out_channels=64,    # Output dimension
            k=12,               # Number of nearest neighbors
            norm='batch',       # Normalization type
            act='relu'          # Activation function
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Get a batch
        for batch in dataloader:
            print("\nBatch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key}: (dict)")
            
            # Get the points and features
            points = batch['points']
            features = batch['features']
            
            print(f"\nRunning forward pass through DeepGCN:")
            print(f"  Input points shape: {points.shape}")
            print(f"  Input features shape: {features.shape}")
            
            # Run a forward pass
            try:
                with torch.no_grad():
                    out, h = model(points, features)
                print(f"  Output shapes: out={out.shape}, h={h.shape}")
                print("\n✅ Successfully ran DeepGCN forward pass!")
            except Exception as e:
                print(f"\n❌ Error running DeepGCN forward pass: {e}")
                import traceback
                traceback.print_exc()
            
            # Only process the first batch
            break
        
    except ImportError as e:
        print(f"Could not import DeepGCN: {e}")
        print("\nTrying alternative approach by copying DeepGCN code...")
        
        # If we can't import directly, try to implement the model here
        try:
            # Define a simplified version of DeepGCN for testing
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            class BasicConv(nn.Module):
                def __init__(self, in_channels, out_channels, activation='relu', norm=None):
                    super().__init__()
                    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                    self.norm = nn.BatchNorm2d(out_channels) if norm == 'batch' else None
                    self.activation = getattr(F, activation) if activation else None
                
                def forward(self, x):
                    x = self.conv(x)
                    if self.norm:
                        x = self.norm(x)
                    if self.activation:
                        x = self.activation(x)
                    return x
            
            class SimpleDeepGCN(nn.Module):
                def __init__(self, in_channels=6, channels=64, out_channels=64):
                    super().__init__()
                    self.input_conv = BasicConv(in_channels, channels, 'relu', 'batch')
                    self.feature_conv = BasicConv(channels, channels, 'relu', 'batch')
                    self.output_conv = BasicConv(channels, out_channels, None, None)
                
                def forward(self, points, features):
                    # Simple forward pass just to test tensor shapes
                    x = self.input_conv(features)
                    x = self.feature_conv(x)
                    x = self.output_conv(x)
                    return x, x
            
            print("Created a simplified DeepGCN model for testing")
            
            # Create the model
            model = SimpleDeepGCN(
                in_channels=6,
                channels=64,
                out_channels=64
            )
            
            # Get a batch
            for batch in dataloader:
                print("\nBatch shapes:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"  {key}: (dict)")
                
                # Get the points and features
                points = batch['points']
                features = batch['features']
                
                print(f"\nRunning forward pass through simplified DeepGCN:")
                print(f"  Input points shape: {points.shape}")
                print(f"  Input features shape: {features.shape}")
                
                # Run a forward pass
                try:
                    with torch.no_grad():
                        out, h = model(points, features)
                    print(f"  Output shapes: out={out.shape}, h={h.shape}")
                    print("\n✅ Successfully ran simplified DeepGCN forward pass!")
                except Exception as e:
                    print(f"\n❌ Error running simplified DeepGCN forward pass: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Only process the first batch
                break
                
        except Exception as e:
            print(f"Could not create simplified DeepGCN: {e}")

if __name__ == "__main__":
    main() 