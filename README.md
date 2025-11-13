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

Supervised Alignment Mode:
Align images with or without using a neural network and user inputs
Interactive 3D visualization for precise adjustments
Correction UI for fine-tuning transformations

Unsupervised Alignment Mode:
Apply a pre-trained model to new data
Batch process multiple samples
Support for all registration types (channel, drift, jitter, etc.)

Supervised Training Mode:
Train a new registration model with user guidance
Data augmentation capabilities
Progress tracking and model validation

Registration Net Architecture:
The neural network architecture consists of several key components:
Feature Extractor:
Multi-level feature extraction with residual connections and progressive downsampling:

Level 0: Initial downsampling (1/2) with 16 channels
Level 1: Moderate expansion (1/4) with 32 channels
Level 2: Intermediate expansion with 64 channels
Level 3: Balanced growth (1/8) with 128 channels
Level 4: Deep features (1/16) with 256 channels

Feature Delta:
Computes transformation parameters between corresponding features:

Channel attention mechanism for feature weighting:
Sequential rotation computation in Z-Y-X order
Translation estimation across multiple feature levels

Transformation Heads:
Final transformation prediction networks:
Translation prediction network
Rotation prediction network

Loss Function:
Specialized loss function for rigid transformation:

MSE for rotation and translation components
Rotation penalty for large angles

Advanced Features:
Data Augmentation
Enhance training data with synthetic transformations:

Random rotations (configurable angle range)
Random translations (configurable magnitude)
Augmentation factor controls quantity of synthetic samples

Training Dynamics
Comprehensive tracking of training progress:

Loss trajectories
Learning rate adaptation
Epoch timing
Model checkpointing

Interactive Visualization
VTK-based 3D visualization with:

Multi-planar viewing (XY, YZ, XZ)
Independent brightness/contrast controls
Registration quality assessment tools

Core Classes
RegistrationNet: Main neural network model for registration
TrainingWorker: Handles model training workflow
RegistrationWorker: Performs image registration using trained models
FileSorting: Organizes files based on naming conventions
VisualizationData: Container for visualization parameters
ImagePair: Container for pairs of images to be registered

Settings Classes
TrainingSettings: Configuration for model training
RegistrationSettings: Configuration for image registration
SettingsFactory: Creates appropriate settings instances

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
