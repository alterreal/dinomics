"""dinomics: DINOv2/DINOv3 feature extraction for medical images."""

from dinomics.aggregate import aggregate_patches, aggregate_slices
from dinomics.config import DinomicsConfig, load_config
from dinomics.extraction import DinoExtractor, FeatureResult, extract_features
from dinomics.preprocess import crop_centered_on_mask, pad_to_square
from dinomics.visualize import pca_patch_map, plot_pca_features

__version__ = "0.1.0"

__all__ = [
    "DinoExtractor",
    "DinomicsConfig",
    "FeatureResult",
    "aggregate_patches",
    "aggregate_slices",
    "crop_centered_on_mask",
    "extract_features",
    "load_config",
    "pad_to_square",
    "pca_patch_map",
    "plot_pca_features",
    "__version__",
]
