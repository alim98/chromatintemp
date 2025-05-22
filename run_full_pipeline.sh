#!/bin/bash
# run_full_pipeline.sh - Comprehensive script to run the full MorphoFeatures pipeline
# Usage: bash run_full_pipeline.sh [options]

set -e  # Exit on error

# Set up Python path for MorphoFeatures
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/MorphoFeatures

# # Add a line to install MorphoFeatures if needed
# if [ ! -d "MorphoFeatures/morphofeatures.egg-info" ]; then
#   echo "Installing MorphoFeatures..."
#   (cd MorphoFeatures && pip install -e .)
# fi

# ====== CONFIGURATION ======
# Default values (can be overridden with command line arguments)
DATA_ROOT="nuclei_sample_1a_v1"
LOW_RES_DIR="low_res_dataset"  # Added low-res dataset directory
OUTPUT_DIR="results/morphofeatures"
BATCH_SIZE=8
NUM_WORKERS=4
EPOCHS=1
MODEL_TYPE="all"  # Options: nucleus, texture-lowres, texture-highres, shape, all
GPU_ID=0
PRECOMPUTED_DIR="data/mesh_cache"
CLASS_CSV="chromatin_classes_and_samples.csv"  # Adding class CSV parameter
USE_WANDB=true

# ====== PARSE ARGUMENTS ======
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data-root)
            DATA_ROOT="$2"
            shift
            shift
            ;;
        --class-csv)  # Adding command line option for class CSV
            CLASS_CSV="$2"
            shift
            shift
            ;;
        --low-res-dir)  # New option for low-res dataset
            LOW_RES_DIR="$2"
            shift
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift
            shift
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift
            shift
            ;;
        --precomputed-dir)
            PRECOMPUTED_DIR="$2"
            shift
            shift
            ;;
        --no-wandb)
            USE_WANDB=false
            shift
            ;;
        --help)
            echo "Usage: bash run_full_pipeline.sh [options]"
            echo "Options:"
            echo "  --data-root DIR        Root directory containing data samples (default: $DATA_ROOT)"
            echo "  --class-csv FILE       CSV file with chromatin classes and samples (default: $CLASS_CSV)"
            echo "  --low-res-dir DIR      Directory with low-resolution data (default: $LOW_RES_DIR)"
            echo "  --output-dir DIR       Output directory for results (default: $OUTPUT_DIR)"
            echo "  --batch-size N         Batch size for training (default: $BATCH_SIZE)"
            echo "  --num-workers N        Number of data loader workers (default: $NUM_WORKERS)"
            echo "  --epochs N             Number of training epochs (default: $EPOCHS)"
            echo "  --model-type TYPE      Model type to train - nucleus, texture-lowres, texture-highres, shape, all (default: $MODEL_TYPE)"
            echo "  --gpu-id ID            GPU ID to use (default: $GPU_ID)"
            echo "  --precomputed-dir DIR  Directory for precomputed/cached meshes (default: $PRECOMPUTED_DIR)"
            echo "  --no-wandb             Disable Weights & Biases logging"
            echo "  --help                 Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ====== PRINT CONFIGURATION ======
echo "===== MORPHOFEATURES PIPELINE CONFIGURATION ====="
echo "Data root: $DATA_ROOT"
echo "Class CSV: $CLASS_CSV"
echo "Low-res dataset: $LOW_RES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"
echo "Epochs: $EPOCHS"
echo "Model type: $MODEL_TYPE"
echo "GPU ID: $GPU_ID"
echo "Precomputed directory: $PRECOMPUTED_DIR"
echo "Using W&B: $USE_WANDB"
echo "==============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PRECOMPUTED_DIR/meshes"
mkdir -p "$PRECOMPUTED_DIR/pointclouds"

# ====== STEP 1: GENERATE MESHES IF NEEDED ======
echo "[1/3] Generating meshes if missing..."
python generate_meshes_if_missing.py \
    --root-dir "$DATA_ROOT" \
    --class-csv "chromatin_classes_and_samples.csv" \
    --cache-dir "$PRECOMPUTED_DIR" \
    --num-workers "$NUM_WORKERS"

# ====== STEP 2: TRAIN MODELS ======
echo "[2/3] Training models..."

# Construct wandb flag
WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAG="--use-wandb"
fi

# Run appropriate training based on model type
case $MODEL_TYPE in
    nucleus)
        echo "WARNING: Nucleus model training from command line is not implemented."
        echo "Consider using 'shape' or 'texture-lowres', 'texture-highres', or 'all' for all models."
        exit 1
        ;;
    texture-lowres)
        python train_texture_with_morphofeatures.py \
            --project_dir "$OUTPUT_DIR" \
            --texture_type "coarse" \
            --config "configs/lowres_texture_config.yaml" \
            --devices "$GPU_ID" \
            --from_checkpoint 0
        ;;
    texture-highres)
        python train_texture_with_morphofeatures.py \
            --project_dir "$OUTPUT_DIR" \
            --texture_type "fine" \
            --config "configs/highres_texture_config.yaml" \
            --devices "$GPU_ID" \
            --from_checkpoint 0
        ;;
    texture)
        echo "Please specify texture resolution: texture-lowres or texture-highres"
        echo "Falling back to texture-lowres..."
        python train_texture_with_morphofeatures.py \
            --project_dir "$OUTPUT_DIR" \
            --texture_type "coarse" \
            --config "configs/lowres_texture_config.yaml" \
            --devices "$GPU_ID" \
            --from_checkpoint 0
        ;;
    shape)
        python train_morphofeatures_models.py \
            --data-root "$DATA_ROOT" \
            --low-res-dir "$LOW_RES_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --model-type shape \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --epochs "$EPOCHS" \
            --gpu-id "$GPU_ID" \
            --precomputed-dir "$PRECOMPUTED_DIR" \
            $WANDB_FLAG
        ;;
    all)
        # # Train shape model first
        # python train_morphofeatures_models.py \
        #     --data-root "$DATA_ROOT" \
        #     --low-res-dir "$LOW_RES_DIR" \
        #     --output-dir "$OUTPUT_DIR" \
        #     --model-type shape \
        #     --batch-size "$BATCH_SIZE" \
        #     --num-workers "$NUM_WORKERS" \
        #     --epochs "$EPOCHS" \
        #     --gpu-id "$GPU_ID" \
        #     --precomputed-dir "$PRECOMPUTED_DIR" \
        #     $WANDB_FLAG
            
        # # Then train low-res texture model
        # python train_texture_with_morphofeatures.py \
        #     --project_dir "$OUTPUT_DIR" \
        #     --texture_type "coarse" \
        #     --config "configs/lowres_texture_config.yaml" \
        #     --devices "$GPU_ID" \
        #     --from_checkpoint 0
            
        # Finally train high-res texture model
        python train_texture_with_morphofeatures.py \
            --project_dir "$OUTPUT_DIR" \
            --texture_type "fine" \
            --config "configs/highres_texture_config.yaml" \
            --devices "$GPU_ID" \
            --from_checkpoint 0
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        echo "Valid options: nucleus, texture-lowres, texture-highres, shape, all"
        exit 1
        ;;
esac

# ====== STEP 3: GENERATE EMBEDDINGS ======
echo "[3/3] Generating embeddings..."
# Format the device string properly for PyTorch (cuda:0 or cpu)
if [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    # If GPU_ID is a number, format as cuda:N
    DEVICE="cuda:$GPU_ID"
else
    # Otherwise, use as-is (might be 'cpu' or already formatted)
    DEVICE="$GPU_ID"
fi

python generate_embeddings.py \
    --root_dir "$DATA_ROOT" \
    --class_csv "$CLASS_CSV" \
    --output_dir "$OUTPUT_DIR/embeddings" \
    --nucleus_shape_model "$OUTPUT_DIR/shape_model.pt" \
    --nucleus_coarse_texture_model "$OUTPUT_DIR/coarse_texture_model.pt" \
    --nucleus_fine_texture_model "$OUTPUT_DIR/fine_texture_model.pt" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE"

echo "===== PIPELINE COMPLETED SUCCESSFULLY ====="
echo "Results saved to: $OUTPUT_DIR"
echo "Embeddings saved to: $OUTPUT_DIR/embeddings" 