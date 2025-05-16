#!/bin/bash
# Script to train all MorphoFeatures models sequentially

# Default values
BATCH_SIZE=2
EPOCHS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--batch-size N] [--epochs N]"
      exit 1
      ;;
  esac
done

echo "Starting MorphoFeatures model training pipeline..."
echo "Using batch size: $BATCH_SIZE, epochs: $EPOCHS"

# Function to modify config file with new batch size and epochs
update_config() {
    local config_file=$1
    local tmp_file="${config_file}.tmp"
    
    # Update batch size in loader section
    sed "s/batch_size: [0-9]*/batch_size: $BATCH_SIZE/g" "$config_file" > "$tmp_file"
    
    # Update epochs in training section
    sed -i "s/epochs: [0-9]*/epochs: $EPOCHS/g" "$tmp_file"
    
    # Replace original file
    mv "$tmp_file" "$config_file"
    echo "Updated config file $config_file with batch size $BATCH_SIZE and epochs $EPOCHS"
}

# Update all config files
update_config "configs/shape_config.yaml"
update_config "configs/lowres_texture_config.yaml"
update_config "configs/highres_texture_config.yaml"

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