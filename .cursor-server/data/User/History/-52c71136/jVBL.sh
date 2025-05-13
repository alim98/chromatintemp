#!/bin/bash

# Uninstall problem packages
pip uninstall -y scipy scikit-image

# Install specific versions compatible with the system
pip install scipy==1.8.0 scikit-image==0.19.3

# Check if installation was successful
python -c "import scipy; print(f'scipy version: {scipy.__version__}')"
python -c "import skimage; print(f'scikit-image version: {skimage.__version__}')"

# Try to run the training script with --help
echo "Attempting to run train_morphofeatures_models.py with --help..."
python train_morphofeatures_models.py --help 