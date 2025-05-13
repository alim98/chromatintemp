
# VGG3D Fine-Tuning Pipeline

This project implements a pipeline for fine-tuning a 3D convolutional neural network (VGG3D) on volumetric nuclei data for classification of chromatin structures. Here's a comprehensive breakdown of the pipeline:

## 1. Data Organization and Processing

### Dataset Structure
- The data is organized in a hierarchical directory structure with each sample in its own folder (numbered from 0001 to 0500+)
- Each sample directory contains:
  - `raw/`: Directory containing raw 3D image slices (TIFF files)
  - `mask/`: Directory containing corresponding mask slices

### Data Loading Process
1. The pipeline uses a custom `NucleiDataset` class (in `dataloader/nuclei_dataloader.py`) that implements true lazy loading
2. The dataset is initialized with:
   - A root directory containing the sample folders
   - Optional class CSV file specifying class labels for each sample
   - Filtering options (by class ID, sample percentage, etc.)
   - Target and crop size dimensions

3. Sample Processing:
   - The dataloader scans the directory structure to identify available samples
   - For each valid sample, it identifies dimensions and possible crops
   - It tracks metadata (sample ID, class, dimensions) without loading actual data yet
   - Only the required volumes are loaded when accessed via `__getitem__`

4. Data Cropping:
   - 3D volumes are cropped into smaller patches of size `crop_size` (default 80x80x80)
   - The cropping strategy divides each sample into non-overlapping patches
   - For a large sample, multiple crops can be extracted which increases the dataset size

5. Efficient Loading:
   - The `_load_volume_slice` method only loads the specific slices needed for a given crop
   - A caching mechanism (`volume_cache`) stores recently accessed volumes to reduce I/O
   - The cache is managed to prevent memory issues by limiting size to 10 samples

## 2. Data Transformation and Augmentation

The pipeline includes several transformation functions for preparing the data:

### Basic Transform (`basic_transform`)
1. Converts input to numpy array if not already
2. If margin is specified, discards pixels from all edges (helps reduce edge artifacts)
3. Converts to PyTorch tensor and normalizes to [0, 1] range
4. Reshapes to 5D tensor with shape (1, 1, D, H, W) where:
   - First dimension is batch size (1)
   - Second dimension is channels (1 for grayscale)
   - Remaining dimensions are depth, height, width
5. Handles various input shapes and includes error checking

### Augmented Transform (`augmented_transform`)
Applies the basic transform plus data augmentation techniques:
1. Random flips along each axis (x, y, z) with 50% probability
2. Random rotations in the xy plane (90°, 180°, 270°) with 50% probability
3. Random intensity adjustments:
   - Gamma correction (factor between 0.8-1.2)
   - Contrast adjustment (factor between 0.8-1.2)
4. Random noise addition (up to 5% level) with 30% probability

### Mask Transform
Similar to basic transform but specifically for segmentation masks, preserving their values.

## 3. Model Architecture

### Base VGG3D Model
The VGG3D model is defined in `model/vgg3d.py` and consists of:

1. **Feature Extractor**: 
   - A series of 3D convolutional blocks
   - Each block has multiple Conv3D layers followed by BatchNorm and ReLU
   - MaxPool3D layer at the end of each block for downsampling
   - The number of feature maps increases progressively (typically doubling)

2. **Classifier**:
   - Fully connected layers that process the flattened feature maps
   - Default configuration: 4096 → 4096 → output_classes
   - Includes ReLU activations and dropout (0.5) for regularization

### Adapter Architecture for Fine-tuning

The script uses adapter models to handle potential mismatches between the pretrained model and the target task:

1. **Vgg3DAdapter/CustomVgg3DAdapter**:
   - Reuses the feature extractor from the base model
   - Creates a new classifier that matches the output size of the feature extractor
   - Allows for reduced classifier size (specified by `classifier_size` parameter)
   - The custom adapter determines the correct feature size by running a forward pass with a sample input

2. **Feature Size Calculation**:
   - The script tries to get a real sample from the dataloader
   - If that fails, it creates a synthetic sample with the target dimensions
   - It runs this sample through the feature extractor to get the exact number of features
   - This ensures the input-output dimensions of the model are compatible

## 4. Training Process

### Model Initialization
1. Loads the pretrained VGG3D model from checkpoint
2. Creates an adapter model with the appropriate classifier size
3. Freezes feature extractor layers if `freeze_features=True` (default)
4. Optionally, selectively freezes classifier layers
5. Sets up optimizer (Adam or SGD) with specified learning rate
6. Configures learning rate scheduler (Cosine Annealing or ReduceLROnPlateau)

### Training Loop
1. For each epoch:
   - **Progressive Unfreezing** (if enabled):
     - Initially all feature layers are frozen
     - Progressively unfreezes groups of layers every 3 epochs
     - Allows gradual adaptation of the pretrained feature extractor

   - **Training Phase**:
     - Processes batches with error handling for shape/NaN issues
     - Optionally uses mixed precision training (with `amp.autocast()` and scaler)
     - Performs forward pass, loss calculation, backward pass, and optimization
     - Tracks training metrics (loss, accuracy) for each batch

   - **Validation Phase**:
     - Runs at specified frequency (e.g., every epoch or every N epochs)
     - Evaluates model on validation data with error handling
     - Calculates validation metrics (loss, accuracy, F1 score)
     - Can force CPU validation to avoid CUDA memory issues

   - **Model Saving**:
     - Saves checkpoints based on best metrics (loss, accuracy, or F1)
     - Implements early stopping if no improvement for N epochs
     - Updates learning rate using the scheduler

   - **Memory Management**:
     - Clears CUDA cache periodically
     - Handles out-of-memory errors gracefully
     - Implements monitoring of system and GPU memory if requested

### Evaluation
After training, the model is evaluated with enhanced metrics:
1. Standard metrics (accuracy, F1 score)
2. Confusion matrix (both raw and normalized)
3. Per-class precision, recall, and F1 scores
4. Top-2 accuracy for multi-class problems
5. Visualization of results (plots and reports)

## 5. Key Features and Optimizations

### Memory and Performance Optimizations
1. **Lazy Loading**: Only loads necessary data when needed
2. **Caching**: Keeps recently accessed volumes in memory
3. **Mixed Precision Training**: Uses 16-bit floating point for faster training
4. **Error Handling**: Robust handling of shape issues, NaN values, and CUDA errors
5. **Memory Monitoring**: Optional real-time monitoring of system and GPU memory

### Training Enhancements
1. **Progressive Unfreezing**: Gradually adapts pretrained features
2. **Early Stopping**: Prevents overfitting by stopping when validation metrics plateau
3. **Learning Rate Scheduling**: Adapts learning rate during training
4. **Data Augmentation**: Increases effective dataset size and improves generalization
5. **Margin Parameter**: Reduces edge artifacts by discarding edge pixels

### Fine-tuning Controls
1. **Selective Layer Freezing**: Control which layers to freeze during training
2. **Reduced Classifier**: Option to use smaller classifier layers to reduce parameters
3. **Class Filtering**: Train on specific class subsets
4. **Sample Percentage**: Control how much of the dataset to use
5. **Batch Limiting**: Restrict batches processed per epoch for debugging

## 6. Command-line Arguments

The script provides extensive control through command-line arguments, categorized as:
1. **Data parameters**: Paths, filtering, subsampling options
2. **Model parameters**: Architecture, checkpoint, target size, output classes
3. **Training parameters**: Batch size, epochs, learning rate, optimizers, regularization
4. **Enhancement options**: Schedulers, mixed precision, progressive unfreezing, augmentation
5. **Debugging options**: Memory monitoring, CPU forcing, CUDA blocking settings

## 7. Data Flow in the Pipeline

1. Data is loaded from disk using the nuclei dataloader
2. A class CSV file determines the class labels for each sample
3. Data is cropped and transformed into the proper tensor format
4. Batches are processed through the model with forward/backward passes
5. The model is periodically validated and saved based on performance
6. After training, comprehensive evaluation metrics are calculated
7. Visualizations and result files are saved to the output directory

