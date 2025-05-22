import numpy as np
import torch
import torch.nn as nn
import os

class Vgg3D(nn.Module):
    def __init__(
        self,
        input_size=(80, 80, 80),
        fmaps=24,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
        fmap_inc=(2, 2, 2, 2),
        n_convolutions=(4, 2, 2, 2),
        output_classes=7,
        input_fmaps=1,
    ):
        super(Vgg3D, self).__init__()

        if len(downsample_factors) != len(fmap_inc):
            raise ValueError("fmap_inc needs to have same length as downsample factors")
        if len(n_convolutions) != len(fmap_inc):
            raise ValueError("n_convolutions needs to have the same length as downsample factors")
        if np.any(np.array(n_convolutions) < 1):
            raise ValueError("Each layer must have at least one convolution")

        current_fmaps = input_fmaps
        current_size = np.array(input_size)

        layers = []
        for i, (df, nc) in enumerate(zip(downsample_factors, n_convolutions)):
            layers += [
                nn.Conv3d(current_fmaps, fmaps, kernel_size=3, padding=1),
                nn.BatchNorm3d(fmaps),
                nn.ReLU(inplace=True)
            ]

            for _ in range(nc - 1):
                layers += [
                    nn.Conv3d(fmaps, fmaps, kernel_size=3, padding=1),
                    nn.BatchNorm3d(fmaps),
                    nn.ReLU(inplace=True)
                ]

            layers.append(nn.MaxPool3d(df))

            current_fmaps = fmaps
            fmaps *= fmap_inc[i]

            current_size = np.floor(current_size / np.array(df))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(int(np.prod(current_size)) * current_fmaps, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_classes),
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        if return_features:
            return x
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model_from_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint, handling potential size mismatches in the final classifier layer.
    
    Args:
        model: The model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        The model with loaded weights
    """
    print(f"Starting to load model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return model
        
    print(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / (1024*1024*1024):.2f} GB")
    
    # Check available memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"Available CUDA memory: {free_mem / (1024*1024*1024):.2f} GB")
    
    try:
        print("Loading checkpoint to CPU first...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint loaded to CPU successfully")
        
        print("Checkpoint contents:")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dictionary with keys: {list(checkpoint.keys())}")
            # Extract the state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            if 'model_state_dict' in checkpoint:
                print("Found 'model_state_dict' key in checkpoint")
                print(f"Loading state_dict with {len(checkpoint['model_state_dict'])} layers...")
                state_dict = checkpoint['model_state_dict']
            else:
                print("Using checkpoint directly as state_dict")
                print(f"Loading state_dict with {len(checkpoint)} layers...")
                state_dict = checkpoint
                
            # Check for classifier layer size mismatch
            output_layer_weight_key = 'classifier.6.weight'
            output_layer_bias_key = 'classifier.6.bias'
            
            if (output_layer_weight_key in state_dict and 
                state_dict[output_layer_weight_key].size(0) != model.classifier[-1].weight.size(0)):
                
                # Size mismatch in output layer
                checkpoint_classes = state_dict[output_layer_weight_key].size(0)
                model_classes = model.classifier[-1].weight.size(0)
                
                print(f"Output layer size mismatch: checkpoint has {checkpoint_classes} classes, model has {model_classes} classes")
                print("Loading all layers except the final classification layer")
                
                # Remove the mismatched layers from the state dict
                state_dict.pop(output_layer_weight_key, None)
                state_dict.pop(output_layer_bias_key, None)
                
                # Load the rest of the layers
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded pretrained weights for all compatible layers")
                print(f"Final classification layer with {model_classes} classes initialized randomly")
            else:
                # No mismatch or not checking for it, load normally
                model.load_state_dict(state_dict)
                print(f"Model loaded successfully from {checkpoint_path}")
        else:
            print(f"Unexpected checkpoint type: {type(checkpoint)}")
            
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return model