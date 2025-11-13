"""
Align - A 3D image registration package with neural network support.

This package provides tools for aligning 3D microscopy images using
both neural network-based approaches and manual user inputs.
"""

from .registration_net import RegistrationNet, TransformLoss, FeatureExtractor, FeatureDelta, TransformationHeads
from .settings import TrainingSettings, RegistrationSettings, SettingsFactory, VisualizationData, ImagePair
from .engine import RegistrationWorker, get_metadata
from .file_sorting import FileSorting, retrieve_counts
from .train import TrainingWorker

__version__ = "0.1.0"
__author__ = "Matthias Blanc"
__email__ = "matthias.blnc@gmail.com"

__all__ = [
    # Core registration model
    'RegistrationNet',
    'TransformLoss',
    'FeatureExtractor',
    'FeatureDelta',
    'TransformationHeads',
    
    # Settings and configuration
    'TrainingSettings',
    'RegistrationSettings',
    'SettingsFactory',
    'VisualizationData',
    'ImagePair',
    
    # Processing engines
    'RegistrationWorker',
    'TrainingWorker',
    
    # Utilities
    'FileSorting',
    'retrieve_counts',
    'get_metadata',
]