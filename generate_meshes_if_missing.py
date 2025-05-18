#!/usr/bin/env python3
"""
Python script to check for existing meshes and generate them if missing, using the same logic as the shell script.
"""
import os
import csv
from utils.mesh_utils import create_mesh_from_mask
from utils.pointcloud import load_mask_volume
from tqdm import tqdm
ROOT_DIR = "nuclei_sample_1a_v1"
CLASS_CSV = "chromatin_classes_and_samples_full.csv"
CACHE_DIR = "data/mesh_cache"
MESH_DIR = os.path.join(CACHE_DIR, "meshes")

os.makedirs(MESH_DIR, exist_ok=True)

def is_valid_sample_id(sid,class_id):
    sid = sid.strip()
    unclassified_flag=False
    if class_id == '19':
        unclassified_flag= True
    print(f"Sample ID: {sid}, Class ID: {class_id}, Unclassified Flag: {unclassified_flag}")
    return sid.isdigit() and len(sid) > 0 and not unclassified_flag

def main():
    with open(CLASS_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        for row in tqdm(rows, desc="Processing samples"):
            sample_id = row.get('sample_id', '').strip()
            class_id = row.get('class_id', '').strip()
            if not is_valid_sample_id(sample_id,class_id):
                print(f"Skipping invalid sample ID {sample_id} with class ID {class_id}.")
                continue
            padded_id = f"{int(sample_id):04d}"
            mesh_file = os.path.join(MESH_DIR, f"{padded_id}_mesh.ply")
            if os.path.exists(mesh_file):
                print(f"Mesh already exists for sample {padded_id}.")
                continue
            print(f"Generating mesh for sample {padded_id}...")
            mask_dir = os.path.join(ROOT_DIR, padded_id, 'mask')
            if not os.path.exists(mask_dir):
                print(f"Mask directory does not exist for {padded_id}")
                continue
            try:
                mask_volume = load_mask_volume(mask_dir)
                mesh = create_mesh_from_mask(mask_volume, threshold=0.5, voxel_size=(1.0,1.0,1.0), smooth_iterations=10, decimate_target=5000)
                if mesh:
                    mesh.export(mesh_file)
                    print(f"Mesh exported for sample {padded_id}.")
                    
            except Exception as e:
                print(f"Error processing sample {padded_id}: {e}")

    print("Done checking and generating meshes.")

if __name__ == "__main__":
    main()
