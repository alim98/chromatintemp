import os
import numpy as np
from PIL import Image
import imageio
import torch

from torchvision import transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataloader.highres_image_dataloader import get_highres_image_dataloader


def denormalize(tensor):
    return tensor * 0.229 + 0.485

def batch_sample_to_gif(dataloader, 
                        sample_index=0, 
                        out_path="volume.gif", 
                        duration=0.2):
    """
    Take one sample (all its z_window_size images) from your high-res loader
    and write them into an animated GIF.
    """
    batch = next(iter(dataloader))
    
    imgs = batch['image'][sample_index]  
    meta = batch['metadata']
    sid = meta['sample_id'][sample_index]
    
    frames = []
    
    for z in range(imgs.shape[0]):  
        img = imgs[z].cpu()  
        img = denormalize(img)                   
        img = img.clamp(0,1)                     
        
        npimg = (img.squeeze().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(npimg)
        frames.append(pil)
    
    imageio.mimsave(out_path, frames, format="GIF", duration=duration)
    print(f"Saved GIF for sample {sid} with {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    
    loader = get_highres_image_dataloader(
        root_dir="data/nuclei_sample_1a_v1",
        batch_size=4,
        class_csv_path="chromatin_classes_and_samples.csv",
        target_size=(224,224),  
        sample_percent=20,
        z_window_size=224,  
        debug=False,
        num_workers=0,
        shuffle=False  
    )

    
    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")  
    
    batch_sample_to_gif(loader, sample_index=0, out_path="sample_highres.gif", duration=0.15)
