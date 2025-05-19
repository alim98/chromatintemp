# MorphoFeatures package

# Import major components to make them available at the package level
# This allows imports like: from morphofeatures import TextureNet

# Texture module components
try:
    from .texture import TextureNet, CellLoaders
except ImportError:
    pass  # Optional component

# Shape module components (if available)
try:
    from .shape import DeepGCN
except ImportError:
    pass  # Optional component

# Neural network module components
try:
    from .nn.losses import NTXentLoss, MorphoFeaturesLoss
except ImportError:
    pass  # Optional component
