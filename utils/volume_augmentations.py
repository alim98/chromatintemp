import numpy as np
import scipy.ndimage as ndi
import random

__all__ = [
    'random_flip',
    'random_rotate90',
    'elastic_deformation',
    'compose_augmentations'
]

def random_flip(volume, axes=(0, 1, 2)):
    """Randomly flip the 3D volume along given axes."""
    for ax in axes:
        if random.random() > 0.5:
            volume = np.flip(volume, axis=ax)
    return volume.copy()

def random_rotate90(volume):
    """Random 90-degree rotation around a random axis."""
    k = random.randint(0, 3)
    axis_pairs = [(1, 2), (0, 2), (0, 1)]
    axes = axis_pairs[random.randint(0, 2)]
    return np.rot90(volume, k, axes).copy()

def elastic_deformation(volume, alpha=15, sigma=3):
    """Elastic deformation for 3D volumes (CPU, slow but simple)."""
    assert volume.ndim == 3
    random_state = np.random.RandomState(None)
    shape = volume.shape
    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(z+dz, (-1,)) , np.reshape(y+dy, (-1,)), np.reshape(x+dx, (-1,))
    deformed = ndi.map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)
    return deformed

def compose_augmentations(*funcs):
    """Return a callable that applies all augmentations in sequence."""
    def _augment(vol):
        for f in funcs:
            vol = f(vol)
        return vol
    return _augment 