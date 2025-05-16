#!/bin/bash
# Script to train all MorphoFeatures models sequentially

echo "Starting MorphoFeatures model training pipeline..."

# Train the shape model
echo "Training Shape Model..."
python train_morphofeatures_models.py --model shape --config configs/shape_config.yaml
if [ $? -ne 0 ]; then
    echo "Shape model training failed!"
    exit 1
fi
echo "Shape model training completed successfully."

# Train the low-resolution texture model
echo "Training Low-Resolution Texture Model..."
python train_morphofeatures_models.py --model lowres --config configs/lowres_texture_config.yaml
if [ $? -ne 0 ]; then
    echo "Low-resolution texture model training failed!"
    exit 1
fi
echo "Low-resolution texture model training completed successfully."

# Train the high-resolution texture model
echo "Training High-Resolution Texture Model..."
python train_morphofeatures_models.py --model highres --config configs/highres_texture_config.yaml
if [ $? -ne 0 ]; then
    echo "High-resolution texture model training failed!"
    exit 1
fi
echo "High-resolution texture model training completed successfully."

echo "All models trained successfully!" 