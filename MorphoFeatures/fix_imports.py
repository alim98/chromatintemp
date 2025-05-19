#!/usr/bin/env python
"""
This script fixes the import issues by creating a symlink to the morphofeatures package 
in the Python path. This avoids having to modify all the imports in the codebase.
"""
import os
import sys
import site

def create_symlink():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Create the symlink
    source = os.path.join(current_dir, "morphofeatures")
    target = os.path.join(site_packages, "morphofeatures")
    
    # Check if the target already exists
    if os.path.exists(target):
        if os.path.islink(target):
            print(f"Removing existing symlink: {target}")
            os.unlink(target)
        else:
            print(f"Error: {target} already exists and is not a symlink. Please remove it manually.")
            return False
    
    # Create the symlink
    print(f"Creating symlink: {source} -> {target}")
    os.symlink(source, target)
    
    print("Symlink created successfully!")
    print("You can now import morphofeatures from anywhere in your Python code.")
    
    return True

def add_to_path():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the parent directory to the path
    parent_dir = os.path.dirname(current_dir)
    
    # Check if the directory is already in path
    if parent_dir not in sys.path:
        print(f"Adding {parent_dir} to PYTHONPATH")
        
        # Create or update .pth file in site-packages
        site_packages = site.getsitepackages()[0]
        pth_file = os.path.join(site_packages, "morphofeatures.pth")
        
        with open(pth_file, "w") as f:
            f.write(parent_dir)
        
        print(f"Created {pth_file} with content: {parent_dir}")
    else:
        print(f"{parent_dir} is already in PYTHONPATH")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix import paths for MorphoFeatures")
    parser.add_argument("--method", choices=["symlink", "path", "both"], default="both",
                        help="Method to fix imports: symlink, path, or both")
    
    args = parser.parse_args()
    
    if args.method in ["symlink", "both"]:
        create_symlink()
    
    if args.method in ["path", "both"]:
        add_to_path()
    
    print("Import paths fixed! Please restart any running Python sessions.") 