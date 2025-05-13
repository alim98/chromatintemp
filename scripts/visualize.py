import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from skimage.transform import resize
from PIL import Image

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader


class NucleiVisualizer:
    """
    A class for visualizing nuclei data from the NucleiDataset.
    Supports visualization of both 2D slices and 3D volumes.
    """
    def __init__(self, output_dir=None, cmap='gray', mask_cmap='hot', alpha=0.7, figsize=(12, 10)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to config.VISUALIZATION_OUTPUT_DIR.
            cmap (str): Colormap for raw images.
            mask_cmap (str): Colormap for mask overlays.
            alpha (float): Alpha value for mask overlay transparency.
            figsize (tuple): Figure size for plots.
        """
        self.output_dir = output_dir if output_dir else config.VISUALIZATION_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.cmap = cmap
        self.mask_cmap = mask_cmap
        self.alpha = alpha
        self.figsize = figsize
        
        # Create custom colormaps for mask overlays
        red_cmap = self._create_transparent_cmap('red')
        self.mask_cmaps = {
            'red': red_cmap,
            'blue': self._create_transparent_cmap('blue'),
            'green': self._create_transparent_cmap('green'),
            'hot': plt.cm.hot,
            'binary': plt.cm.binary
        }

    def _create_transparent_cmap(self, color):
        """
        Create a transparent colormap for overlay visualization.
        
        Args:
            color (str): Base color for the colormap.
            
        Returns:
            LinearSegmentedColormap: A matplotlib colormap.
        """
        if color == 'red':
            color_tuple = (1, 0, 0)
        elif color == 'green':
            color_tuple = (0, 1, 0)
        elif color == 'blue':
            color_tuple = (0, 0, 1)
        else:
            color_tuple = (1, 0, 0)  # Default to red
            
        # Create colormap with transparency
        cdict = {
            'red': [(0, 0, 0), (1, color_tuple[0], color_tuple[0])],
            'green': [(0, 0, 0), (1, color_tuple[1], color_tuple[1])],
            'blue': [(0, 0, 0), (1, color_tuple[2], color_tuple[2])],
            'alpha': [(0, 0, 0), (1, self.alpha, self.alpha)]
        }
        
        return LinearSegmentedColormap(f'transparent_{color}', cdict)

    def _tensor_to_numpy(self, tensor):
        """
        Convert a PyTorch tensor to a NumPy array.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            
        Returns:
            np.ndarray: NumPy array.
        """
        if tensor is None:
            return None
            
        if isinstance(tensor, torch.Tensor):
            # Move to CPU if on GPU
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
                
            # Remove batch and channel dimensions if present
            if tensor.ndim > 3:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            if tensor.ndim > 2:
                tensor = tensor.squeeze(0)  # Remove channel dimension
                
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            # Remove batch and channel dimensions if present
            if tensor.ndim > 3:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            if tensor.ndim > 2:
                tensor = tensor.squeeze(0)  # Remove channel dimension
                
            return tensor
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")

    def _normalize_array(self, array):
        """
        Normalize array to [0, 1] for visualization.
        
        Args:
            array (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Normalized array.
        """
        if array is None:
            return None
            
        # Check if already normalized
        if array.min() >= 0 and array.max() <= 1:
            return array
            
        # Handle empty array
        if array.max() == array.min():
            return np.zeros_like(array)
            
        return (array - array.min()) / (array.max() - array.min())

    def visualize_slice(self, data_item, save_path=None, show=True, title=None):
        """
        Visualize a 2D slice from the dataset.
        
        Args:
            data_item (dict): Data item from the dataloader (__getitem__ output).
            save_path (str, optional): Path to save the visualization.
            show (bool): Whether to display the plot.
            title (str, optional): Title for the plot.
            
        Returns:
            tuple: Figure and axes objects.
        """
        # Extract data from data_item
        if 'image' in data_item:
            # 2D slice data
            image = self._tensor_to_numpy(data_item['image'])
            mask = self._tensor_to_numpy(data_item['mask'])
        elif 'volume' in data_item:
            # Extract middle slice from 3D volume
            image = self._tensor_to_numpy(data_item['volume'])
            mask = self._tensor_to_numpy(data_item['mask'])
            if image.ndim == 3:
                mid_idx = image.shape[0] // 2
                image = image[mid_idx]
                if mask is not None:
                    mask = mask[mid_idx]
        else:
            raise ValueError("Invalid data format")
            
        # Normalize arrays
        image_norm = self._normalize_array(image)
        mask_norm = self._normalize_array(mask) if mask is not None else None
        
        # Set up the plot
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Plot raw image
        axes[0].imshow(image_norm, cmap=self.cmap)
        axes[0].set_title("Raw Image")
        axes[0].axis('off')
        
        # Plot mask
        if mask_norm is not None:
            axes[1].imshow(mask_norm, cmap='binary')
            axes[1].set_title("Mask")
            axes[1].axis('off')
        else:
            axes[1].set_visible(False)
        
        # Plot overlay
        if mask_norm is not None:
            axes[2].imshow(image_norm, cmap=self.cmap)
            # Create a more visible mask overlay with red color
            mask_overlay = np.ma.masked_where(mask_norm < 0.5, mask_norm)  # Increase threshold to make only strong mask regions visible
            axes[2].imshow(mask_overlay, cmap='autumn', alpha=0.7, interpolation='none')  # Use 'autumn' colormap which is more visible
            axes[2].set_title("Overlay")
            axes[2].axis('off')
        else:
            axes[2].set_visible(False)
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            fig.suptitle(title)
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            slice_num = metadata.get('slice_num', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            fig.suptitle(f"Sample: {sample_id}, Slice: {slice_num}, Class: {class_name}")
            
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                slice_num = metadata.get('slice_num', '0')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_slice_{slice_num}.png")
                plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes

    def visualize_volume(self, data_item, save_path=None, show=True, title=None, 
                         axis='z', frames=10, interval=200, colorbar=True):
        """
        Visualize a 3D volume from the dataset as an animation.
        
        Args:
            data_item (dict): Data item from the dataloader (__getitem__ output).
            save_path (str, optional): Path to save the animation (as .gif or .mp4).
            show (bool): Whether to display the animation.
            title (str, optional): Title for the animation.
            axis (str): Axis to slice along ('x', 'y', or 'z').
            frames (int): Number of frames to show (0 for all slices).
            interval (int): Interval between frames in ms.
            colorbar (bool): Whether to show a colorbar.
            
        Returns:
            tuple: Figure, axes, and animation objects.
        """
        # Extract data from data_item
        if 'volume' not in data_item:
            raise ValueError("Input data does not contain a volume")
            
        volume = self._tensor_to_numpy(data_item['volume'])
        mask = self._tensor_to_numpy(data_item['mask'])
        
        # Normalize arrays
        volume_norm = self._normalize_array(volume)
        mask_norm = self._normalize_array(mask) if mask is not None else None
        
        # Determine slicing axis
        if axis == 'z':
            n_slices = volume_norm.shape[0]
            get_slice = lambda i: (volume_norm[i], mask_norm[i] if mask_norm is not None else None)
        elif axis == 'y':
            n_slices = volume_norm.shape[1]
            get_slice = lambda i: (volume_norm[:, i, :], mask_norm[:, i, :] if mask_norm is not None else None)
        elif axis == 'x':
            n_slices = volume_norm.shape[2]
            get_slice = lambda i: (volume_norm[:, :, i], mask_norm[:, :, i] if mask_norm is not None else None)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
        
        # Determine frames to show
        if frames <= 0 or frames > n_slices:
            frames = n_slices
            
        step = max(1, n_slices // frames)
        slice_indices = range(0, n_slices, step)[:frames]
        
        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Initialize plots
        img_plots = []
        mask_plots = []
        overlay_plots = []
        
        # Raw image
        img_slice, _ = get_slice(slice_indices[0])
        img_plot = axes[0].imshow(img_slice, cmap=self.cmap)
        axes[0].set_title("Raw Volume")
        axes[0].axis('off')
        img_plots.append(img_plot)
        
        # Add colorbar
        if colorbar:
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img_plot, cax=cax)
        
        # Mask
        if mask_norm is not None:
            _, mask_slice = get_slice(slice_indices[0])
            mask_plot = axes[1].imshow(mask_slice, cmap='binary')
            axes[1].set_title("Mask")
            axes[1].axis('off')
            mask_plots.append(mask_plot)
            
            if colorbar:
                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(mask_plot, cax=cax)
        else:
            axes[1].set_visible(False)
        
        # Overlay
        if mask_norm is not None:
            img_slice, mask_slice = get_slice(slice_indices[0])
            overlay_img = axes[2].imshow(img_slice, cmap=self.cmap)
            masked_data = np.ma.masked_where(mask_slice < 0.5, mask_slice)  # Increased threshold
            overlay_mask = axes[2].imshow(masked_data, cmap='autumn', alpha=0.7, interpolation='none')  # Using autumn colormap
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            overlay_plots.append(overlay_img)
            overlay_plots.append(overlay_mask)
        else:
            axes[2].set_visible(False)
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            main_title = title
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            main_title = f"Sample: {sample_id}, Class: {class_name}"
            
        plt.tight_layout()
        
        # Update function for animation
        def update(i):
            slice_idx = slice_indices[i]
            img_slice, mask_slice = get_slice(slice_idx)
            
            # Update raw image
            img_plots[0].set_array(img_slice)
            
            # Update mask
            if mask_norm is not None and mask_plots:
                mask_plots[0].set_array(mask_slice)
            
            # Update overlay
            if mask_norm is not None and overlay_plots:
                overlay_plots[0].set_array(img_slice)
                masked_data = np.ma.masked_where(mask_slice < 0.5, mask_slice)
                overlay_plots[1].set_array(masked_data)
                
            # Update title with slice number
            fig.suptitle(f"{main_title} - Slice {slice_idx}/{n_slices-1} ({axis}-axis)")
            
            return img_plots + mask_plots + overlay_plots
            
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(slice_indices), 
                                      interval=interval, blit=False)
        # Save animation if requested
        if save_path:
            try:
                # Ensure .gif extension
                if not save_path.endswith('.gif'):
                    save_path += '.gif'
                ani.save(save_path, writer='pillow', fps=1000/interval)
            except Exception as e:
                print(f"Failed to save animation: {e}")
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_volume_{axis}_axis.gif")
                try:
                    ani.save(auto_save_path, writer='pillow', fps=1000/interval)
                except Exception as e:
                    print(f"Failed to save animation: {e}")
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes, ani

    def visualize_batch(self, batch, max_samples=4, save_dir=None, show=True):
        """
        Visualize a batch of samples.
        
        Args:
            batch (dict): Batch of samples from dataloader.
            max_samples (int): Maximum number of samples to visualize.
            save_dir (str, optional): Directory to save visualizations.
            show (bool): Whether to display the visualizations.
            
        Returns:
            list: List of figure objects.
        """
        figures = []
        
        # Determine if we have volumes or 2D images
        if 'volume' in batch:
            is_volume = True
            data_key = 'volume'
            mask_key = 'mask'
        elif 'image' in batch:
            is_volume = False
            data_key = 'image'
            mask_key = 'mask'
        else:
            raise ValueError("Batch does not contain recognizable data format")
            
        # Get batch size - with the custom collate_fn, these are now lists
        batch_size = len(batch[data_key])
            
        # Limit samples to visualize
        n_samples = min(batch_size, max_samples)
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Visualize each sample
        for i in range(n_samples):
            # Extract sample data
            sample = {}
            
            # With custom_collate_fn, everything is already a list
            for key in batch.keys():
                if key == 'metadata':
                    # Handle metadata dictionary special case
                    sample[key] = {k: batch[key][k][i] for k in batch[key]}
                else:
                    # For other keys, just get the i-th element from the list
                    sample[key] = batch[key][i]
            
            # Generate save path if needed
            if save_dir:
                metadata = sample.get('metadata', {})
                sample_id = metadata.get('sample_id', f'sample_{i}')
                if is_volume:
                    save_path = os.path.join(save_dir, f"{sample_id}_volume.png")
                else:
                    slice_num = metadata.get('slice_num', 0)
                    save_path = os.path.join(save_dir, f"{sample_id}_slice_{slice_num}.png")
            else:
                save_path = None
                
            # Call appropriate visualization method
            if is_volume:
                # For volumes, just show the middle slice for batch visualization
                fig, axes = self.visualize_slice(sample, save_path=save_path, show=show)
                figures.append(fig)
            else:
                fig, axes = self.visualize_slice(sample, save_path=save_path, show=show)
                figures.append(fig)
                
        return figures 

    def visualize_resize_comparison(self, data_item, original_volume, original_mask, save_path=None, show=True, title=None):
        """
        Visualize a side-by-side comparison of original and resized volumes.
        
        Args:
            data_item (dict): Data item from the dataloader with resized volume.
            original_volume (np.ndarray): Original volume before resizing.
            original_mask (np.ndarray): Original mask before resizing.
            save_path (str, optional): Path to save the visualization.
            show (bool): Whether to display the plot.
            title (str, optional): Title for the plot.
            
        Returns:
            tuple: Figure and axes objects.
        """
        # Extract resized data
        resized_volume = self._tensor_to_numpy(data_item['volume'])
        resized_mask = self._tensor_to_numpy(data_item['mask'])
        
        # Get middle slices
        orig_mid_idx = original_volume.shape[0] // 2
        orig_img = original_volume[orig_mid_idx]
        orig_mask_slice = original_mask[orig_mid_idx] if original_mask is not None else None
        
        resized_mid_idx = resized_volume.shape[0] // 2
        resized_img = resized_volume[resized_mid_idx]
        resized_mask_slice = resized_mask[resized_mid_idx] if resized_mask is not None else None
        
        # Normalize for visualization
        orig_img_norm = self._normalize_array(orig_img)
        orig_mask_norm = self._normalize_array(orig_mask_slice) if orig_mask_slice is not None else None
        
        resized_img_norm = self._normalize_array(resized_img)
        resized_mask_norm = self._normalize_array(resized_mask_slice) if resized_mask_slice is not None else None
        
        # Create a 2x3 subplot grid (original and resized, each with raw/mask/overlay)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # --- Original volume visualization (top row) ---
        # Raw image
        axes[0, 0].imshow(orig_img_norm, cmap=self.cmap)
        axes[0, 0].set_title("Original - Raw Image")
        axes[0, 0].axis('off')
        
        # Mask
        if orig_mask_norm is not None:
            axes[0, 1].imshow(orig_mask_norm, cmap='binary')
            axes[0, 1].set_title("Original - Mask")
            axes[0, 1].axis('off')
        else:
            axes[0, 1].set_visible(False)
        
        # Overlay
        if orig_mask_norm is not None:
            axes[0, 2].imshow(orig_img_norm, cmap=self.cmap)
            mask_overlay = np.ma.masked_where(orig_mask_norm < 0.5, orig_mask_norm)
            axes[0, 2].imshow(mask_overlay, cmap='autumn', alpha=0.7, interpolation='none')
            axes[0, 2].set_title("Original - Overlay")
            axes[0, 2].axis('off')
        else:
            axes[0, 2].set_visible(False)
            
        # --- Resized volume visualization (bottom row) ---
        # Raw image
        axes[1, 0].imshow(resized_img_norm, cmap=self.cmap)
        axes[1, 0].set_title(f"Resized ({resized_volume.shape[0]}x{resized_volume.shape[1]}x{resized_volume.shape[2]}) - Raw Image")
        axes[1, 0].axis('off')
        
        # Mask
        if resized_mask_norm is not None:
            axes[1, 1].imshow(resized_mask_norm, cmap='binary')
            axes[1, 1].set_title("Resized - Mask")
            axes[1, 1].axis('off')
        else:
            axes[1, 1].set_visible(False)
        
        # Overlay
        if resized_mask_norm is not None:
            axes[1, 2].imshow(resized_img_norm, cmap=self.cmap)
            mask_overlay = np.ma.masked_where(resized_mask_norm < 0.5, resized_mask_norm)
            axes[1, 2].imshow(mask_overlay, cmap='autumn', alpha=0.7, interpolation='none')
            axes[1, 2].set_title("Resized - Overlay")
            axes[1, 2].axis('off')
        else:
            axes[1, 2].set_visible(False)
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            original_shape = metadata.get('original_shape', 'Unknown')
            fig.suptitle(f"Sample: {sample_id}, Class: {class_name}\nOriginal shape: {original_shape}", fontsize=14)
            
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_resize_comparison.png")
                plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes 

    def visualize_resize_comparison_animation(self, data_item, original_volume, original_mask, save_path=None, show=True, title=None,
                                      axis='z', frames=10, interval=200):
        """
        Create an animation that shows original and resized volumes side by side for comparison.
        
        Args:
            data_item (dict): Data item from the dataloader with resized volume.
            original_volume (np.ndarray): Original volume before resizing.
            original_mask (np.ndarray): Original mask before resizing.
            save_path (str, optional): Path to save the animation (as .gif).
            show (bool): Whether to display the animation.
            title (str, optional): Title for the animation.
            axis (str): Axis to slice along ('x', 'y', or 'z').
            frames (int): Number of frames to show (0 for all slices).
            interval (int): Interval between frames in ms.
            
        Returns:
            tuple: Figure, axes, and animation objects.
        """
        # Extract resized data
        resized_volume = self._tensor_to_numpy(data_item['volume'])
        resized_mask = self._tensor_to_numpy(data_item['mask'])
        
        # Normalize arrays
        orig_volume_norm = self._normalize_array(original_volume)
        orig_mask_norm = self._normalize_array(original_mask) if original_mask is not None else None
        
        resized_volume_norm = self._normalize_array(resized_volume)
        resized_mask_norm = self._normalize_array(resized_mask) if resized_mask is not None else None
        
        # Determine slicing axis for original
        if axis == 'z':
            orig_n_slices = original_volume.shape[0]
            orig_get_slice = lambda i: (orig_volume_norm[i], orig_mask_norm[i] if orig_mask_norm is not None else None)
        elif axis == 'y':
            orig_n_slices = original_volume.shape[1]
            orig_get_slice = lambda i: (orig_volume_norm[:, i, :], orig_mask_norm[:, i, :] if orig_mask_norm is not None else None)
        elif axis == 'x':
            orig_n_slices = original_volume.shape[2]
            orig_get_slice = lambda i: (orig_volume_norm[:, :, i], orig_mask_norm[:, :, i] if orig_mask_norm is not None else None)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
            
        # Determine slicing axis for resized
        if axis == 'z':
            resized_n_slices = resized_volume_norm.shape[0]
            resized_get_slice = lambda i: (resized_volume_norm[i], resized_mask_norm[i] if resized_mask_norm is not None else None)
        elif axis == 'y':
            resized_n_slices = resized_volume_norm.shape[1]
            resized_get_slice = lambda i: (resized_volume_norm[:, i, :], resized_mask_norm[:, i, :] if resized_mask_norm is not None else None)
        elif axis == 'x':
            resized_n_slices = resized_volume_norm.shape[2]
            resized_get_slice = lambda i: (resized_volume_norm[:, :, i], resized_mask_norm[:, :, i] if resized_mask_norm is not None else None)
        
        # Calculate normalized slice positions for consistent comparison
        if frames <= 0:
            frames = min(orig_n_slices, resized_n_slices)
            
        # Create a larger figure with 2 rows (original and resized) and 3 columns (raw, mask, overlay)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Calculate slice indices for original and resized volumes
        # We want to show comparable slices (e.g., 25%, 50%, 75% through each volume)
        orig_step = max(1, orig_n_slices // frames)
        orig_slice_indices = list(range(0, orig_n_slices, orig_step))[:frames]
        
        resized_step = max(1, resized_n_slices // frames)
        resized_slice_indices = list(range(0, resized_n_slices, resized_step))[:frames]
        
        # Make sure we have the same number of frames for both
        min_frames = min(len(orig_slice_indices), len(resized_slice_indices))
        orig_slice_indices = orig_slice_indices[:min_frames]
        resized_slice_indices = resized_slice_indices[:min_frames]
        
        # Initialize plots - original row (top)
        # Original Raw
        orig_img, _ = orig_get_slice(orig_slice_indices[0])
        orig_img_plot = axes[0, 0].imshow(orig_img, cmap=self.cmap)
        axes[0, 0].set_title(f"Original ({original_volume.shape[0]}x{original_volume.shape[1]}x{original_volume.shape[2]}) - Raw")
        axes[0, 0].axis('off')
        
        # Original Mask
        if orig_mask_norm is not None:
            _, orig_mask_slice = orig_get_slice(orig_slice_indices[0])
            orig_mask_plot = axes[0, 1].imshow(orig_mask_slice, cmap='binary')
            axes[0, 1].set_title("Original - Mask")
            axes[0, 1].axis('off')
        else:
            axes[0, 1].set_visible(False)
            orig_mask_plot = None
        
        # Original Overlay
        if orig_mask_norm is not None:
            orig_img_slice, orig_mask_slice = orig_get_slice(orig_slice_indices[0])
            orig_overlay_img = axes[0, 2].imshow(orig_img_slice, cmap=self.cmap)
            orig_masked_data = np.ma.masked_where(orig_mask_slice < 0.5, orig_mask_slice)
            orig_overlay_mask = axes[0, 2].imshow(orig_masked_data, cmap='autumn', alpha=0.7, interpolation='none')
            axes[0, 2].set_title("Original - Overlay")
            axes[0, 2].axis('off')
        else:
            axes[0, 2].set_visible(False)
            orig_overlay_img = None
            orig_overlay_mask = None
            
        # Initialize plots - resized row (bottom)
        # Resized Raw
        resized_img, _ = resized_get_slice(resized_slice_indices[0])
        resized_img_plot = axes[1, 0].imshow(resized_img, cmap=self.cmap)
        axes[1, 0].set_title(f"Resized ({resized_volume.shape[0]}x{resized_volume.shape[1]}x{resized_volume.shape[2]}) - Raw")
        axes[1, 0].axis('off')
        
        # Resized Mask
        if resized_mask_norm is not None:
            _, resized_mask_slice = resized_get_slice(resized_slice_indices[0])
            resized_mask_plot = axes[1, 1].imshow(resized_mask_slice, cmap='binary')
            axes[1, 1].set_title("Resized - Mask")
            axes[1, 1].axis('off')
        else:
            axes[1, 1].set_visible(False)
            resized_mask_plot = None
        
        # Resized Overlay
        if resized_mask_norm is not None:
            resized_img_slice, resized_mask_slice = resized_get_slice(resized_slice_indices[0])
            resized_overlay_img = axes[1, 2].imshow(resized_img_slice, cmap=self.cmap)
            resized_masked_data = np.ma.masked_where(resized_mask_slice < 0.5, resized_mask_slice)
            resized_overlay_mask = axes[1, 2].imshow(resized_masked_data, cmap='autumn', alpha=0.7, interpolation='none')
            axes[1, 2].set_title("Resized - Overlay")
            axes[1, 2].axis('off')
        else:
            axes[1, 2].set_visible(False)
            resized_overlay_img = None
            resized_overlay_mask = None
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            main_title = title
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            main_title = f"Sample: {sample_id}, Class: {class_name}"
            
        plt.tight_layout()
        
        # Create lists to store all plot objects that will be updated
        orig_plots = [p for p in [orig_img_plot, orig_mask_plot, orig_overlay_img, orig_overlay_mask] if p is not None]
        resized_plots = [p for p in [resized_img_plot, resized_mask_plot, resized_overlay_img, resized_overlay_mask] if p is not None]
        
        # Update function for animation
        def update(frame_idx):
            # Get slice indices for this frame
            orig_idx = orig_slice_indices[frame_idx]
            resized_idx = resized_slice_indices[frame_idx]
            
            # Update original plots
            orig_img_slice, orig_mask_slice = orig_get_slice(orig_idx)
            orig_img_plot.set_array(orig_img_slice)
            
            if orig_mask_norm is not None and orig_mask_plot is not None:
                orig_mask_plot.set_array(orig_mask_slice)
            
            if orig_mask_norm is not None and orig_overlay_img is not None and orig_overlay_mask is not None:
                orig_overlay_img.set_array(orig_img_slice)
                orig_masked_data = np.ma.masked_where(orig_mask_slice < 0.5, orig_mask_slice)
                orig_overlay_mask.set_array(orig_masked_data)
            
            # Update resized plots
            resized_img_slice, resized_mask_slice = resized_get_slice(resized_idx)
            resized_img_plot.set_array(resized_img_slice)
            
            if resized_mask_norm is not None and resized_mask_plot is not None:
                resized_mask_plot.set_array(resized_mask_slice)
            
            if resized_mask_norm is not None and resized_overlay_img is not None and resized_overlay_mask is not None:
                resized_overlay_img.set_array(resized_img_slice)
                resized_masked_data = np.ma.masked_where(resized_mask_slice < 0.5, resized_mask_slice)
                resized_overlay_mask.set_array(resized_masked_data)
                
            # Update title with slice numbers
            percentage = frame_idx / (len(orig_slice_indices) - 1) * 100 if len(orig_slice_indices) > 1 else 0
            fig.suptitle(f"{main_title} - Original Slice {orig_idx}/{orig_n_slices-1}, Resized Slice {resized_idx}/{resized_n_slices-1} (≈{percentage:.0f}% through volume)")
            
            return orig_plots + resized_plots
            
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(orig_slice_indices), 
                                      interval=interval, blit=False)
        
        # Save animation if requested
        if save_path:
            try:
                # Ensure .gif extension
                if not save_path.endswith('.gif'):
                    save_path += '.gif'
                ani.save(save_path, writer='pillow', fps=1000/interval)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_resize_comparison_animation.gif")
                try:
                    ani.save(auto_save_path, writer='pillow', fps=1000/interval)
                    print(f"Animation saved to {auto_save_path}")
                except Exception as e:
                    print(f"Failed to save animation: {e}")
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes, ani 