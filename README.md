# RAD
A 3D image rigid registration and alignment tool designed for microscopy data, with neural network-based capabilities.

# Overview
RAD is a Python package that provides tools for aligning 3D microscopy images. It features both neural network-based approaches and manual methods, with an interactive GUI for ease of use. The package is especially suited for alignment of samples that are multi-channel and offers experimental options for time-lapse microscopy data.

# Features
Neural Network-Based Registration: Uses a custom architecture with feature extraction and progressive pathway for accurate 3D image alignment

<img width="1172" height="357" alt="image" src="https://github.com/user-attachments/assets/e8af55ef-0ccc-4fac-a2ae-d6c5e2d0ec7c" />
A: Training dataset augmentation
B: Overall convolutional neural network architecture
C: Details of feature extraction layers


# Registration Modes
Channel Alignment: Align different channels within the same sample
<br>Batch Processing: Align multiple samples to a common reference
<br>Drift Correction: Correct for sample drift in time-lapse imaging
<br>Jitter Correction: Stabilize images with reference to a windowed average
<br>Global Reference Alignment: Align all timepoints using a reference frame

# Requirements
Python 3.8+
PyTorch
SimpleITK
PyQt5
VTK
NumPy
tifffile

# File Naming Conventions
The package can handle various file naming patterns using prefix settings:

Sample Prefix: Pattern identifying different samples (e.g., "SPM01", "SPM02")
Time Prefix: Pattern identifying timepoints (e.g., "TM001", "TM002")
Channel Prefix: Pattern identifying channels (e.g., "CHN1", "CHN2")

Example: For files named "SPM01_TM002_CHN3.tif", set:

Sample Prefix: "SPM"
Time Prefix: "TM"
Channel Prefix: "CHN"

# GUI Usage
The graphical user interface provides three main operational modes:

<br>Supervised Alignment Mode:
<br>Align images with or without using a neural network and user inputs
<br>Interactive 3D visualization for precise adjustments
<br>Correction UI for fine-tuning transformations

<br>Unsupervised Alignment Mode:
<br>Apply a pre-trained model to new data
<br>Batch process multiple samples
<br>Support for all registration types (channel, drift, jitter, etc.)

<br>Supervised Training Mode:
<br>Train a new registration model with user guidance
<br>Data augmentation capabilities
<br>Progress tracking and model validation

<br>Registration Net Architecture:
<br>The neural network architecture consists of several key components:
<br>Feature Extractor:
<br>Multi-level feature extraction with residual connections and progressive downsampling:

<br>Level 0: Initial downsampling (1/2) with 16 channels
<br>Level 1: Moderate expansion (1/4) with 32 channels
<br>Level 2: Intermediate expansion with 64 channels
<br>Level 3: Balanced growth (1/8) with 128 channels
<br>Level 4: Deep features (1/16) with 256 channels

<br>Feature Delta:
<br>Computes transformation parameters between corresponding features:

<br>Channel attention mechanism for feature weighting:
<br>Sequential rotation computation in Z-Y-X order
<br>Translation estimation across multiple feature levels

<br>Transformation Heads:
<br>Final transformation prediction networks:
<br>Translation prediction network
<br>Rotation prediction network

<br>Loss Function:
<br>Specialized loss function for rigid transformation:

<br>MSE for rotation and translation components
<br>Rotation penalty for large angles

<br>Advanced Features:
<br>Data Augmentation
<br>Enhance training data with synthetic transformations:

<br>Random rotations (configurable angle range)
<br>Random translations (configurable magnitude)
<br>Augmentation factor controls quantity of synthetic samples

<br>Training Dynamics
<br>Comprehensive tracking of training progress:

<br>Loss trajectories
<br>Learning rate adaptation
<br>Epoch timing
<br>Model checkpointing

<br>Interactive Visualization
<br>VTK-based 3D visualization with:

<br>Multi-planar viewing (XY, YZ, XZ)
<br>Independent brightness/contrast controls
<br>Registration quality assessment tools

# Core Classes
<br>RegistrationNet: Main neural network model for registration
<br>TrainingWorker: Handles model training workflow
<br>RegistrationWorker: Performs image registration using trained models
<br>FileSorting: Organizes files based on naming conventions
<br>VisualizationData: Container for visualization parameters
<br>ImagePair: Container for pairs of images to be registered

# Settings Classes
<br>TrainingSettings: Configuration for model training
<br>RegistrationSettings: Configuration for image registration
<br>SettingsFactory: Creates appropriate settings instances

# Troubleshooting
Common issues and their solutions:

CUDA Out of Memory: Reduce batch size or window size
File Not Found: Check file naming patterns and prefix settings
Poor Alignment Quality: Try adjusting learning rate or augmentation settings
GUI Crashes: Check for PyQt5 and VTK compatibility

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

This package uses SimpleITK for image manipulation and transformation
PyQt5 and VTK are used for the visualization components
PyTorch provides the neural network framework

# Citation
If you use this software in your research, please cite:
@software{RAD,
  author = {Matthias Blanc},
  title = {RAD: 3D Rigid Alignment with Deep learning},
  year = {2025},
  url = {https://github.com/cristinapujades/Blanc_et_al_2025/tree/main/RAD}
}
