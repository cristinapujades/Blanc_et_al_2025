import os
import time
import datetime
import traceback
import tempfile
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import tifffile
from registration_net import RegistrationNet
from engine import RegistrationWorker
from file_sorting import FileSorting

class AnalysisWorker(QThread):
    """Worker thread for model analysis operations."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.stop_requested = False
        
        # Initialize device for model operations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store model and output paths as class variables
        self.model_path = Path(settings.model_path)
        self.output_path = Path(settings.output_path)
        
        # Initialize path for PDF output
        self.base_filename = self.model_path.stem
        
        # Initialize variables for storing model and metadata
        self.model = None
        self.metadata = None
        
        # Create registration worker for transform operations
        self.registration_worker = None
    
    def run(self):
        """Main execution method for the analysis thread."""
        try:
            self.progress_signal.emit(f"Starting analysis in {self.settings.analysis_mode} mode")

            # Create registration worker (for apply_transform and other utilities)
            self.registration_worker = RegistrationWorker(self.settings)
            
            # Load model and metadata
            self.model, self.metadata = self._load_model()
            
            # Execute appropriate analysis mode
            if self.settings.analysis_mode == 'training_dynamics':
                self._run_training_dynamics()
            elif self.settings.analysis_mode == 'evaluate':
                self._run_evaluation()
            
        except Exception as e:
            self.progress_signal.emit(f"Error during analysis: {str(e)}")
            self.progress_signal.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()
            
    def _load_model(self):
        """Load model and extract metadata."""
        self.progress_signal.emit(f"Loading model")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            model = RegistrationNet().to(self.device)
            
            # Extract state dict based on format
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    metadata = checkpoint.get('metadata', {})
                else:
                    # Legacy format where dict is directly state dict
                    model.load_state_dict(checkpoint)
                    metadata = {}
            else:
                # Direct state dict format
                model.load_state_dict(checkpoint)
                metadata = {}
                
            model.eval()
            
            # If metadata is empty, generate basic info
            if not metadata:
                self.progress_signal.emit("No metadata found, generating basic information...")
                metadata = self._generate_basic_metadata(model)
                
            return model, metadata
            
        except Exception as e:
            self.progress_signal.emit(f"Error loading model: {str(e)}")
            raise
    
    def _generate_basic_metadata(self, model):
        """Generate basic metadata for models without it."""
        return {
            'parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024),
            'timestamp': datetime.datetime.now().isoformat(),
            'note': 'Basic metadata generated during analysis (not original training metadata)',
        }
    
    def load_image_for_model(self, file_path):
        """Load and normalize image for model prediction."""
        if self.registration_worker:
            return self.registration_worker.load_image_for_model(file_path)
        
        # Fallback implementation if RegistrationWorker is not available
        img = tifffile.imread(str(file_path))
        
        # Convert to float32 and normalize to 0-1
        img_norm = img.astype(np.float32)
        if img.dtype == np.uint8:
            img_norm = img_norm / 255.0
        elif img.dtype == np.uint16:
            img_norm = img_norm / 65535.0 
            
        # Convert to tensor and preprocess dimensions
        tensor = torch.from_numpy(img_norm)
        if tensor.dim() > 5:
            tensor = tensor.squeeze()
            
        # If 3D image [D, H, W], add batch and channel dimensions
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        # If 4D image [1, D, H, W], add channel dimension
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    
    def find_test_files(self):
        """Select appropriate files for testing based on processing mode and prefix settings.
        Returns a list of (moving_path, ref_path) tuples ready for evaluation.
        """
        self.progress_signal.emit("Finding test files based on processing mode...")
        
        # Use FileSorting to organize files based on prefixes
        file_sorter = FileSorting(self.settings)
        organized_files = file_sorter.organize_files()
        
        if not organized_files:
            self.progress_signal.emit("No files found based on current prefix settings")
            return []
            
        test_file_pairs = []
        ref_path = Path(self.settings.reference_file) if self.settings.reference_file else None
        
        # Select files based on processing mode
        if self.settings.p_mode in ['batch', 'global']:
            self.progress_signal.emit(f"Using {self.settings.p_mode} mode to select test files")
            
            # For batch mode, use the reference channel from each sample
            for sample_id, timepoints in organized_files.items():
                # Get first timepoint (or reference timepoint in global mode)
                if self.settings.p_mode == 'global' and hasattr(self.settings, 'reference_timepoint'):
                    if self.settings.reference_timepoint == 'First':
                        timepoint_id = min(timepoints.keys())
                    else:  # LAST
                        timepoint_id = max(timepoints.keys())
                else:
                    timepoint_id = next(iter(timepoints.keys()))
                
                # Get the reference channel files from this sample/timepoint
                channels = timepoints[timepoint_id]
                if self.settings.reference_channel in channels:
                    for moving_path in channels[self.settings.reference_channel]:
                        if ref_path:
                            test_file_pairs.append((moving_path, ref_path))
        
        elif self.settings.p_mode == 'channel':
            self.progress_signal.emit("Using channel mode to select test files")
            
            # For channel mode, pick non-reference channels to align against reference channel
            for sample_id, timepoints in organized_files.items():
                timepoint_id = next(iter(timepoints.keys()))
                channels = timepoints[timepoint_id]
                
                # Get reference channel file
                if self.settings.reference_channel in channels:
                    ref_channel_path = channels[self.settings.reference_channel][0] if channels[self.settings.reference_channel] else None
                    
                    # Get all non-reference channels
                    for channel_id, files in channels.items():
                        if channel_id != self.settings.reference_channel and files and ref_channel_path:
                            for moving_path in files:
                                test_file_pairs.append((moving_path, ref_channel_path))
        
        elif self.settings.p_mode == 'drift':
            self.progress_signal.emit("Using drift mode to select test files")
            
            # For drift mode, use consecutive timepoints
            for sample_id, timepoints in organized_files.items():
                # Get sorted timepoints
                sorted_timepoints = sorted(timepoints.keys())
                
                # Process pairs of consecutive timepoints
                for i in range(len(sorted_timepoints) - 1):
                    t1, t2 = sorted_timepoints[i], sorted_timepoints[i + 1]
                    ref_files = timepoints[t1][self.settings.reference_channel]
                    moving_files = timepoints[t2][self.settings.reference_channel]
                    
                    if ref_files and moving_files:
                        for moving_path in moving_files:
                            test_file_pairs.append((moving_path, ref_files[0]))
        
        elif self.settings.p_mode == 'jitter':
            self.progress_signal.emit("Using jitter mode to select test files")
            
            # For jitter mode, we'd typically use the average stack as reference
            if ref_path:
                for sample_id, timepoints in organized_files.items():
                    sorted_timepoints = sorted(timepoints.keys())
                    
                    # If window settings are available, use them
                    if hasattr(self.settings, 'window_size') and hasattr(self.settings, 'ref_timepoint'):
                        half_window = self.settings.window_size // 2
                        ref_pos = sorted_timepoints.index(self.settings.ref_timepoint) if self.settings.ref_timepoint in sorted_timepoints else len(sorted_timepoints) // 2
                        window_start = max(0, ref_pos - half_window)
                        window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)
                        
                        # Use timepoints outside the window
                        test_timepoints = [t for i, t in enumerate(sorted_timepoints) 
                                        if i < window_start or i >= window_end]
                    else:
                        # Without window settings, use all timepoints
                        test_timepoints = sorted_timepoints
                    
                    # Add all relevant files
                    for timepoint_id in test_timepoints:
                        if self.settings.reference_channel in timepoints[timepoint_id]:
                            for moving_path in timepoints[timepoint_id][self.settings.reference_channel]:
                                test_file_pairs.append((moving_path, ref_path))
        
        # Limit test pairs based on the analysis mode
        if self.settings.analysis_mode == 'training_dynamics':
            # For training dynamics, we only need the first pair
            test_file_pairs = test_file_pairs[:1]
            self.progress_signal.emit("Using only the first image pair for feature visualization")
        else:  # 'evaluate' mode
            # For evaluation, use the first 10 pairs
            test_file_pairs = test_file_pairs[:10]
            self.progress_signal.emit(f"Found {len(test_file_pairs)} test file pairs for evaluation (limited to 10)")
        
        return test_file_pairs

    def _plot_training_summary_table(self, ax, td):
        # Extract relevant data
        loss_trajectory = td.get('loss_trajectory', [])
        lr_history = td.get('lr_history', [])
        lr_reduction_epochs = td.get('lr_reduction_epochs', [])
        avg_epoch_time = td.get('avg_epoch_time_seconds')
        
        # Clear axis and remove frame
        ax.axis('off')
        
        if not loss_trajectory:
            ax.text(0.5, 0.5, "No training summary data available", 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate summary statistics
        initial_loss = loss_trajectory[0] if loss_trajectory else 0
        final_loss = loss_trajectory[-1] if loss_trajectory else 0
        best_loss = min(loss_trajectory) if loss_trajectory else 0
        best_epoch = loss_trajectory.index(best_loss) + 1 if loss_trajectory else 0
        improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
        
        # Calculate time-based metrics
        total_epochs = len(loss_trajectory)
        if avg_epoch_time:
            total_training_time = float(avg_epoch_time) * total_epochs
            # Format time as hours:minutes:seconds
            hours, remainder = divmod(total_training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            time_str = "Unknown"
        
        # Calculate convergence metrics
        early_loss = loss_trajectory[min(4, len(loss_trajectory)-1)] if len(loss_trajectory) > 4 else initial_loss
        early_improvement = (initial_loss - early_loss) / initial_loss * 100 if initial_loss > 0 else 0
        late_improvement = (early_loss - final_loss) / early_loss * 100 if early_loss > 0 else 0
        
        # Create table data - Left column
        left_data = [
            ["Initial Loss", f"{initial_loss:.6f}"],
            ["Final Loss", f"{final_loss:.6f}"],
            ["Best Loss", f"{best_loss:.6f}"],
            ["Best Epoch", f"{best_epoch}"],
            ["Total Epochs", f"{total_epochs}"],
            ["Overall Improvement", f"{improvement:.2f}%"],
            ["Early Improvement (First 5)", f"{early_improvement:.2f}%"],
            ["Late Improvement (After 5)", f"{late_improvement:.2f}%"],
        ]
        
        # Create table data - Right column
        right_data = [
            ["Initial LR", f"{lr_history[0]:.6f}" if lr_history else "Unknown"],
            ["Final LR", f"{lr_history[-1]:.6f}" if lr_history else "Unknown"],
            ["LR Reductions", f"{len(lr_reduction_epochs)}"],
            ["Avg Epoch Time", f"{float(avg_epoch_time):.2f}s" if avg_epoch_time else "Unknown"],
            ["Total Training Time", time_str],
            ["Convergence Ratio", f"{final_loss/best_loss:.2f}x"],
        ]
        
        # Add model hardware info if available
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            right_data.append(["Hardware", gpu_info])
        else:
            right_data.append(["Hardware", "CPU"])
        
        # Add metadata if available
        if hasattr(self, 'metadata'):
            if 'processing_mode' in self.metadata:
                right_data.append(["Processing Mode", self.metadata['processing_mode']])
                
        # Create and style the left table (Loss metrics)
        left_table = ax.table(cellText=left_data, loc='center', cellLoc='center',
                            colWidths=[0.6, 0.3], bbox=[0.05, 0.1, 0.45, 0.8])
        left_table.auto_set_font_size(False)
        left_table.set_fontsize(9)
        
        # Create and style the right table (Time & LR metrics)
        right_table = ax.table(cellText=right_data, loc='center', cellLoc='center',
                            colWidths=[0.5, 0.5], bbox=[0.55, 0.1, 0.4, 0.8])
        right_table.auto_set_font_size(False)
        right_table.set_fontsize(9)
        
        # Style both tables
        for table in [left_table, right_table]:
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('black')
                if col == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f0f0f0')
    
    def analyze_architecture(self):
        """Analyze model architecture in detail."""
        model_summary = []
        total_params = 0
        trainable_params = 0
        
        # Analyze each module
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear, torch.nn.Sequential)):
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                if params > 0:  # Only include non-empty modules
                    model_summary.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'params': params,
                        'trainable': trainable
                    })
                    
                    total_params += params
                    trainable_params += trainable
        
        return {
            'layers': model_summary,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'file_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }
    
    def visualize_registration_batch(self, test_file_pairs):
        if not test_file_pairs or len(test_file_pairs) == 0:
            self.progress_signal.emit("No test file pairs provided")
            return None
            
        if not self.registration_worker:
            self.progress_signal.emit("No registration worker available")
            return None
            
        reg_viz_batch = []
        inference_times = []
        
        for idx, (moving_path, ref_path) in enumerate(test_file_pairs):
            moving_path = Path(moving_path)
            ref_path = Path(ref_path)
            
            try:
                # Process this sample
                self.progress_signal.emit(f"Processing pair {idx+1}/{len(test_file_pairs)}: {moving_path.name}")
                
                # Load images using registration worker's method
                reference = self.registration_worker.load_image_for_model(ref_path)
                moving = self.registration_worker.load_image_for_model(moving_path)
                
                # Get raw data for visualization (original dimensions)
                ref_data = reference.squeeze().numpy()
                mov_data = moving.squeeze().numpy()
                
                # Create temporary files for transformed image and resampled moving image
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_out:
                    temp_output_path = temp_out.name
                
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_resampled:
                    temp_resampled_path = temp_resampled.name
                
                # First, apply identity transform to resample moving image to reference space
                identity_transform = torch.zeros(6)
                self.registration_worker.apply_transform(
                    input_path=str(moving_path), 
                    reference_path=str(ref_path),
                    transform=identity_transform,
                    output_path=temp_resampled_path
                )
                
                # Load the resampled moving image
                resampled_data = tifffile.imread(temp_resampled_path).astype(np.float32)
                if resampled_data.dtype == np.uint8:
                    resampled_data = resampled_data / 255.0
                elif resampled_data.dtype == np.uint16:
                    resampled_data = resampled_data / 65535.0
                
                # Process with model
                with torch.no_grad():
                    reference = reference.to(self.device)
                    moving = moving.to(self.device)
                    start_time = time.time()
                    predicted_transform = self.model(reference, moving)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    # Calculate time in milliseconds
                    inference_time = (end_time - start_time) * 1000
                    inference_times.append(inference_time)
                    predicted_transform = predicted_transform.cpu().numpy()[0]
                
                # Apply the predicted transformation
                self.registration_worker.apply_transform(
                    input_path=str(moving_path),
                    reference_path=str(ref_path),
                    transform=torch.tensor(predicted_transform),
                    output_path=temp_output_path
                )
                
                # Load the transformed image
                transformed_data = tifffile.imread(temp_output_path).astype(np.float32)
                if transformed_data.dtype == np.uint8:
                    transformed_data = transformed_data / 255.0
                elif transformed_data.dtype == np.uint16:
                    transformed_data = transformed_data / 65535.0
                
                # Calculate similarity metrics
                def calculate_similarity_metrics(ref_mask, transformed_mask):
                    intersection = np.sum(ref_mask * transformed_mask)
                    ref_sum = np.sum(ref_mask)
                    transformed_sum = np.sum(transformed_mask)
                    
                    # Dice coefficient
                    dice = (2.0 * intersection) / (ref_sum + transformed_sum) if (ref_sum + transformed_sum) > 0 else 0
                    
                    # Jaccard coefficient (IoU)
                    union = ref_sum + transformed_sum - intersection
                    jaccard = intersection / union if union > 0 else 0
                    
                    return dice, jaccard

                # Create binary masks using thresholding
                ref_mask = (ref_data > np.mean(ref_data)).astype(int)
                resampled_mask = (resampled_data > np.mean(resampled_data)).astype(int)
                transformed_mask = (transformed_data > np.mean(transformed_data)).astype(int)

                # Calculate metrics
                dice_before, jaccard_before = calculate_similarity_metrics(ref_mask, resampled_mask)
                dice_after, jaccard_after = calculate_similarity_metrics(ref_mask, transformed_mask)
                
                # Create visualization data
                reg_viz = {
                    'reference': ref_data,
                    'moving': mov_data,
                    'resampled': resampled_data,
                    'transformed': transformed_data,
                    'transform_params': predicted_transform,
                    'transform_rotation': predicted_transform[0:3],
                    'transform_translation': predicted_transform[3:6],
                    'inference_time': inference_time,
                    'dice_before': dice_before,
                    'dice_after': dice_after,
                    'jaccard_before': jaccard_before,
                    'jaccard_after': jaccard_after,
                    'filename': moving_path.name,
                    'ref_filename': ref_path.name
                }
                
                reg_viz_batch.append(reg_viz)
                    
                # Clean up temporary files
                try:
                    os.unlink(temp_output_path)
                    os.unlink(temp_resampled_path)
                except Exception:
                    pass
                
            except Exception as e:
                self.progress_signal.emit(f"Error processing {moving_path.name}: {str(e)}")
                self.progress_signal.emit(traceback.format_exc())
                
        # Calculate average and standard deviation
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            # Store this information for later use
            self.avg_inference_time = avg_inference_time
            self.std_inference_time = std_inference_time

        return reg_viz_batch

    def visualize_features(self, reference_path, moving_path):
        """Extract and visualize features based on weights and activations."""
        self.progress_signal.emit(f"Extracting features ...")
        ref_path = Path(reference_path)
        mov_path = Path(moving_path)
                
        try:
            # Load images
            reference = self.load_image_for_model(ref_path)
            moving = self.load_image_for_model(mov_path)
            
            reference = reference.to(self.device)
            moving = moving.to(self.device)
            
            # Get feature maps using hooks
            feature_maps = {'reference': {}, 'moving': {}}
            hooks = []
            
            def hook_fn(name, img_type):
                def hook(module, input, output):
                    feature_maps[img_type][name] = output.detach().cpu()
                return hook
                
            # Register hooks for feature extractor outputs at all levels
            hook_points = {
                'level1': self.model.progressive_extractor.level1_norm2,
                'level2': self.model.progressive_extractor.level2_norm2,
                'level3': self.model.progressive_extractor.level3_norm2,
                'level4': self.model.progressive_extractor.level4_norm2
            }
            
            # Get feature maps for reference image
            for level_name, module in hook_points.items():
                hooks.append(module.register_forward_hook(hook_fn(level_name, 'reference')))
            
            _ = self.model.progressive_extractor(reference)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Get feature maps for moving image
            hooks = []
            for level_name, module in hook_points.items():
                hooks.append(module.register_forward_hook(hook_fn(level_name, 'moving')))
            
            _ = self.model.progressive_extractor(moving)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Calculate channel importance based on weights
            channel_weights = {}
            
            # Extract weights from the convolution layers in each level
            weight_extraction_points = {
                'level1': self.model.progressive_extractor.level1_conv2,
                'level2': self.model.progressive_extractor.level2_conv2,
                'level3': self.model.progressive_extractor.level3_conv2,
                'level4': self.model.progressive_extractor.level4_conv2
            }
            
            # Calculate importance for each level and channel
            for level_name, conv_layer in weight_extraction_points.items():
                # Get the weights (output_channels, input_channels, kernel_x, kernel_y, kernel_z)
                weights = conv_layer.weight.detach().cpu()
                
                # Calculate L1 norm across input channels and kernel dimensions
                importance = torch.sum(torch.abs(weights), dim=(1, 2, 3, 4)).numpy()
                
                # Store importance scores for this level
                channel_weights[level_name] = importance
            
            # Prepare visualizations
            feature_visualizations = []
            
            # For each level, select top channels by weight and show activations
            for level_name in hook_points.keys():
                if level_name not in channel_weights or level_name not in feature_maps['reference'] or level_name not in feature_maps['moving']:
                    continue
                
                # Get importance weights for channels at this level
                weights = channel_weights[level_name]
                
                # Get feature maps
                ref_features = feature_maps['reference'][level_name]
                mov_features = feature_maps['moving'][level_name]
                
                # Sort channels by weight importance
                sorted_channels = np.argsort(-weights)  # Descending order
                
                # Take top channels by weight
                top_channels = sorted_channels[:10].tolist()
                
                # Add visualizations for top channels
                for img_type, features in [('reference', ref_features), ('moving', mov_features)]:
                    for channel in top_channels:
                        if channel >= features.shape[1]:
                            continue
                            
                        # Get feature data
                        feature_data = features[0, channel].numpy()
                        
                        # Create projections
                        if len(feature_data.shape) == 3:
                            max_xy = np.max(feature_data, axis=0)
                        else:
                            max_xy = feature_data
                        
                        # Calculate activation and weight
                        activation = np.mean(np.abs(feature_data))
                        weight = weights[channel]
                        
                        feature_visualizations.append({
                            'image_type': img_type,
                            'level': level_name,
                            'channel': channel,
                            'activation': activation,
                            'weight': float(weight),
                            'projections': {'xy': max_xy}
                        })
            
            self.progress_signal.emit(f"Extracted {len(feature_visualizations)} feature visualizations across {len(hook_points)} model levels")
            return feature_visualizations
            
        except Exception as e:
            self.progress_signal.emit(f"Error visualizing features: {str(e)}")
            self.progress_signal.emit(traceback.format_exc())
            return None

    def generate_training_page(self, pdf):
        """Generate the first page with model architecture information."""
        plt.figure(figsize=(8.5, 11))
        
        # Title and spacing
        plt.figtext(0.5, 0.95, 'RegistrationNet Model Analysis', 
                    fontsize=18, fontweight='bold', ha='center')
        plt.figtext(0.5, 0.92, f'Model: {self.model_path.name}', 
                    fontsize=12, ha='center')
        
        # Set up grid for side-by-side info panels and schematic
        gs = plt.GridSpec(5, 2, height_ratios=[0, 0.08, 0.25, 0.25, 0.2], hspace=0.6)
        
        # Training info (left column)
        ax1 = plt.subplot(gs[1, 0])
        ax1.axis('off')
        ax1.set_title('Training Information', fontsize=12, fontweight='bold', y=1.0)
        
        train_info = []
        for key in ['total_epochs', 'best_loss', 'final_loss', 'cycles_completed', 
        'training_time_seconds', 'processing_mode', 'trained_from_scratch',
        'n_original_pairs', 'n_augmented_pairs']:
            if key in self.metadata:
                if key == 'training_time_seconds' and self.metadata[key]:
                    hours = self.metadata[key] // 3600
                    minutes = (self.metadata[key] % 3600) // 60
                    seconds = self.metadata[key] % 60
                    value = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                else:
                    value = str(self.metadata[key])
                train_info.append(f"{key.replace('_', ' ').title()}: {value}")
                
        train_text = '\n'.join(train_info) if train_info else "No training metadata available"
        ax1.text(0.05, 0.95, train_text, fontsize=9, 
                va='top', family='monospace', transform=ax1.transAxes)
        
        # Model info (right column)
        ax2 = plt.subplot(gs[1, 1])
        ax2.axis('off')
        ax2.set_title('Model Information', fontsize=12, fontweight='bold', y=1.0)
        
        # Calculate architecture info
        arch_info = self.analyze_architecture()
        
        model_info = [
            f"Parameter Count: {arch_info['trainable_params']:,}",
            f"Model Size: {arch_info['file_size_mb']:.2f} MB"
        ]
        
        # Add hardware info if available
        if torch.cuda.is_available():
            model_info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            model_info.append("Device: CPU")

        # Add timestamp
        model_info.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        model_text = '\n'.join(model_info)
        ax2.text(0.05, 0.95, model_text, fontsize=9, va='top', family='monospace', transform=ax2.transAxes)

        # Check if we have training dynamics data for plots
        if 'training_dynamics' in self.metadata:
            td = self.metadata['training_dynamics']
            
            # Loss trajectory plot
            if 'loss_trajectory' in td or 'loss_trajectories' in td:
                ax3 = plt.subplot(gs[2, :])
                
                # Handle different metadata formats
                if 'loss_trajectory' in td:
                    loss_data = td['loss_trajectory']
                    
                    # Plot the loss trajectory
                    x = range(1, len(loss_data) + 1)
                    ax3.plot(x, loss_data, color='#1f77b4', marker='o', 
                            markersize=3, linewidth=1.5, alpha=0.9, label='Loss')
                    
                    # Mark the lowest loss point
                    if loss_data:
                        min_loss = min(loss_data)
                        min_idx = loss_data.index(min_loss)
                        ax3.scatter([min_idx + 1], [min_loss], s=100, 
                                marker='*', color='gold', edgecolors='black', 
                                linewidth=1, zorder=11, label='Best Loss')
                elif 'loss_trajectories' in td:
                    # Get the first cycle's data
                    cycles = sorted([int(k) for k in td['loss_trajectories'].keys()])
                    if cycles:
                        loss_data = td['loss_trajectories'][str(cycles[0])]
                        
                        # Plot the loss trajectory
                        x = range(1, len(loss_data) + 1)
                        ax3.plot(x, loss_data, color='#1f77b4', marker='o', 
                                markersize=3, linewidth=1.5, alpha=0.9, label=f'Cycle {cycles[0]} Loss')
                        
                        # Mark the lowest loss point
                        if loss_data:
                            min_loss = min(loss_data)
                            min_idx = loss_data.index(min_loss)
                            ax3.scatter([min_idx + 1], [min_loss], s=100, 
                                    marker='*', color='gold', edgecolors='black', 
                                    linewidth=1, zorder=11, label='Best Loss')
                
                # Format the plot
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training Loss Curve')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend()
                
                # Set log scale if range is large
                if loss_data and max(loss_data) / min(loss_data) > 100:
                    ax3.set_yscale('log')
            
            # LR history plot
            if 'lr_history' in td or 'lr_histories' in td:
                ax4 = plt.subplot(gs[3, :])
                
                # Handle different metadata formats
                if 'lr_history' in td:
                    lr_data = td['lr_history']
                    
                    # Plot the LR history
                    x = range(1, len(lr_data) + 1)
                    ax4.plot(x, lr_data, color='#ff7f0e', marker='o', 
                            markersize=3, linewidth=1.5, alpha=0.9, label='Learning Rate')
                
                elif 'lr_histories' in td:
                    # Get the first cycle's data
                    cycles = sorted([int(k) for k in td['lr_histories'].keys()])
                    if cycles:
                        lr_data = td['lr_histories'][str(cycles[0])]
                        
                        # Plot the LR history
                        x = range(1, len(lr_data) + 1)
                        ax4.plot(x, lr_data, color='#ff7f0e', marker='o', 
                                markersize=3, linewidth=1.5, alpha=0.9, label=f'Cycle {cycles[0]} LR')
                
                # Format the plot
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.set_title('Learning Rate Schedule')
                ax4.grid(True, linestyle='--', alpha=0.7)
                ax4.set_yscale('log')  # LR is usually shown in log scale
                ax4.legend()
        else:
            # If no training dynamics data, show a message
            ax3 = plt.subplot(gs[2:4, :])
            ax3.text(0.5, 0.5, "No training dynamics data available in model metadata", 
                    ha='center', va='center', fontsize=14)
            ax3.axis('off')

        ax5 = plt.subplot(gs[4, :])
        self._plot_training_summary_table(ax5, td)

        plt.subplots_adjust(top=0.90, bottom=0.05, left=0.1, right=0.9)
        pdf.savefig()
        plt.close()

    def generate_registration_results_page(self, pdf, reg_viz_batch):
        if not reg_viz_batch or len(reg_viz_batch) == 0:
            return
        
        # Calculate metrics across all samples
        avg_dice_before = np.mean([viz['dice_before'] for viz in reg_viz_batch])
        avg_dice_after = np.mean([viz['dice_after'] for viz in reg_viz_batch])
        avg_jaccard_before = np.mean([viz['jaccard_before'] for viz in reg_viz_batch])
        avg_jaccard_after = np.mean([viz['jaccard_after'] for viz in reg_viz_batch])
        
        # Calculate improvement percentages
        dice_improvement = ((avg_dice_after - avg_dice_before) / max(1e-6, 1 - avg_dice_before)) * 100
        jaccard_improvement = ((avg_jaccard_after - avg_jaccard_before) / max(1e-6, 1 - avg_jaccard_before)) * 100
        
        # Get inference time metrics
        if hasattr(self, 'avg_inference_time') and hasattr(self, 'std_inference_time'):
            avg_inference_time = self.avg_inference_time
            inference_time_std = self.std_inference_time
            inference_text = f"Average Inference Time:\n{avg_inference_time:.2f} ± {inference_time_std:.2f} ms"
        else:
            inference_text = "Average Inference Time:\nNot measured"
        
        # Add hardware info
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            inference_text += f"\nHardware: {gpu_info}"
        else:
            inference_text += "\nHardware: CPU"
        
        # Create figure
        plt.figure(figsize=(8.5, 11))
        plt.suptitle('Registration Performance Analysis', fontsize=16, fontweight='bold', y=0.97)
        
        # Maximum intensity projection helper
        def get_projection(volume):
            if len(volume.shape) == 3:
                return np.max(volume, axis=0)
            return volume
        
        # Get total number of samples (up to 10)
        n_samples = min(len(reg_viz_batch), 10)
        
        # First grid (samples 1-5)
        first_grid_samples = min(5, n_samples)
        if first_grid_samples > 0:
            # Create grid in top half
            gs1 = gridspec.GridSpec(4, first_grid_samples, 
                                top=0.90, 
                                bottom=0.55,
                                left=0.07, 
                                right=0.95,
                                wspace=0.1, 
                                hspace=0.1)

            # Add vertical labels
            plt.figtext(0.05, 0.86, "Moving", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.77, "Aligned", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.68, "Before", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.59, "After", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')

            # Process first 5 samples
            for col in range(first_grid_samples):
                reg_viz = reg_viz_batch[col]
                sample_number = col + 1
                
                # Get projections
                ref_proj = get_projection(reg_viz['reference'])
                mov_proj = get_projection(reg_viz['moving'])
                resampled_proj = get_projection(reg_viz['resampled'])
                transformed_proj = get_projection(reg_viz['transformed'])
                
                # Row 1: Moving image
                ax1 = plt.subplot(gs1[0, col])
                ax1.imshow(mov_proj, cmap='gray')
                ax1.set_title(f"Sample {sample_number}", fontsize=8)
                ax1.axis('off')
                
                # Row 2: Aligned image
                ax2 = plt.subplot(gs1[1, col])
                ax2.imshow(transformed_proj, cmap='gray')
                ax2.axis('off')
                
                # Row 3: Overlay before alignment
                ax3 = plt.subplot(gs1[2, col])
                ax3.imshow(ref_proj, cmap='gray')
                ax3.imshow(resampled_proj, alpha=0.6, cmap='hot')
                ax3.axis('off')
                
                # Row 4: Overlay after alignment
                ax4 = plt.subplot(gs1[3, col])
                ax4.imshow(ref_proj, cmap='gray')
                ax4.imshow(transformed_proj, alpha=0.6, cmap='hot')
                ax4.axis('off')

        # Second grid (samples 6-10)
        second_grid_samples = max(0, min(5, n_samples - 5))
        if second_grid_samples > 0:
            # Create grid in bottom half
            gs2 = gridspec.GridSpec(4, second_grid_samples, 
                                top=0.50, 
                                bottom=0.15,
                                left=0.07, 
                                right=0.95,
                                wspace=0.1, 
                                hspace=0.1)

            # Add vertical labels for second grid
            plt.figtext(0.05, 0.46, "Moving", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.37, "Aligned", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.28, "Before", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')
            plt.figtext(0.05, 0.19, "After", fontsize=8, rotation=90, 
                        fontweight='bold', color='black', ha='center', va='center')

            # Process samples 6-10
            for col in range(second_grid_samples):
                reg_viz = reg_viz_batch[col + 5]
                sample_number = col + 6
                
                # Get projections
                ref_proj = get_projection(reg_viz['reference'])
                mov_proj = get_projection(reg_viz['moving'])
                resampled_proj = get_projection(reg_viz['resampled'])
                transformed_proj = get_projection(reg_viz['transformed'])
                
                # Row 1: Moving image
                ax1 = plt.subplot(gs2[0, col])
                ax1.imshow(mov_proj, cmap='gray')
                ax1.set_title(f"Sample {sample_number}", fontsize=8)
                ax1.axis('off')

                # Row 2: Aligned image
                ax2 = plt.subplot(gs2[1, col])
                ax2.imshow(transformed_proj, cmap='gray')
                ax2.axis('off')
                
                # Row 3: Overlay before alignment
                ax3 = plt.subplot(gs2[2, col])
                ax3.imshow(ref_proj, cmap='gray')
                ax3.imshow(resampled_proj, alpha=0.6, cmap='hot')
                ax3.axis('off')
                
                # Row 4: Overlay after alignment
                ax4 = plt.subplot(gs2[3, col])
                ax4.imshow(ref_proj, cmap='gray')
                ax4.imshow(transformed_proj, alpha=0.6, cmap='hot')
                ax4.axis('off')

        # Add metrics table at the bottom
        metrics_ax = plt.axes([0.1, 0.04, 0.45, 0.08])
        metrics_ax.axis('off')
        
        metrics_table = []
        headers = ['Metric', 'Before', 'After', 'Improvement']
        
        metrics_table.append([
            'Avg Dice', 
            f"{avg_dice_before:.4f}", 
            f"{avg_dice_after:.4f}", 
            f"{dice_improvement:.1f}%"])
        
        metrics_table.append([
            'Avg Jaccard', 
            f"{avg_jaccard_before:.4f}", 
            f"{avg_jaccard_after:.4f}", 
            f"{jaccard_improvement:.1f}%"])
        
        # Create table
        table = plt.table(cellText=metrics_table, colLabels=headers, loc='center', cellLoc='center')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Add inference time text box
        inference_ax = plt.axes([0.63, 0.04, 0.25, 0.08])
        inference_ax.axis('off')
                
        # Create text box with border
        inference_ax.text(0.5, 0.5, inference_text,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                    facecolor='#f0f0f0',
                                    edgecolor='black',
                                    alpha=0.8),
                        fontsize=9, fontweight='normal')
        
        pdf.savefig()
        plt.close()

    def generate_feature_visualization_page(self, pdf, feature_viz):
        """Generate the feature visualization page showing the most important channels."""
        if not feature_viz or len(feature_viz) == 0:
            return
            
        self.progress_signal.emit("Generating feature visualization page...")
        
        # Create a single figure for all levels
        plt.figure(figsize=(8.5, 11))
        plt.suptitle('Feature Projections by Level', fontsize=16, fontweight='bold', y=0.96)
        
        # Group features by level and image type
        features_by_level = {}
        for feat in feature_viz:
            level = feat['level']
            img_type = feat['image_type']
            if level not in features_by_level:
                features_by_level[level] = {'reference': [], 'moving': []}
            features_by_level[level][img_type].append(feat)
        
        # Ensure we have specific order: level1, level2, level3, level4
        level_order = ['level1', 'level2', 'level3', 'level4']
        levels = [level for level in level_order if level in features_by_level]
        
        # Create a more compact grid with 8 rows (2 per level) and 4 columns
        gs = plt.GridSpec(8, 4, left=0.05, hspace=0.1, wspace=0, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1])
        
        # Add column headers
        plt.figtext(0.29, 0.90, 'Reference Features', fontsize=12, fontweight='bold', ha='center')
        plt.figtext(0.75, 0.90, 'Moving Features', fontsize=12, fontweight='bold', ha='center')
        
        # Process each level
        for level_idx, level in enumerate(levels):
            # Fixed row positions based on level
            row_start = level_idx * 2 
            
            # Add level label aligned with the middle of the two rows
            y_pos = 0.78 - (row_start * 0.105)  # Adjusted for 8-row grid
            plt.figtext(0.04, y_pos, level, fontsize=10, fontweight='bold', rotation=90, va='center')
            
            # Get all features for this level to extract the unique channels by weight
            all_level_features = []
            for img_type in ['reference', 'moving']:
                if img_type in features_by_level[level]:
                    all_level_features.extend(features_by_level[level][img_type])
            
            # Get unique channels and their weights
            channels_by_weight = {}
            for feat in all_level_features:
                channel = feat['channel']
                if channel not in channels_by_weight:
                    channels_by_weight[channel] = feat['weight']
            
            # Sort channels by weight
            sorted_channels = sorted(channels_by_weight.keys(), 
                                key=lambda ch: channels_by_weight[ch], 
                                reverse=True)
            
            # Select top 4 channels by weight
            top_channels = sorted_channels[:4]
            
            # Filter features to show only these top channels
            ref_features = [f for f in features_by_level[level]['reference'] 
                        if f['channel'] in top_channels]
            mov_features = [f for f in features_by_level[level]['moving'] 
                        if f['channel'] in top_channels]
            
            # Sort features by channel to align reference and moving
            ref_features.sort(key=lambda x: top_channels.index(x['channel']))
            mov_features.sort(key=lambda x: top_channels.index(x['channel']))
            
            # Plot reference features (left 2 columns)
            for i, feat in enumerate(ref_features):
                # Calculate position: i=0,1 -> row 0, i=2,3 -> row 1
                row = row_start + (i // 2)
                col = i % 2  # 0 or 1
                
                ax = plt.subplot(gs[row, col])
                
                channel = feat['channel']
                activation = feat['activation']
                weight = feat['weight']
                
                # Get XY projection
                projection = feat['projections']['xy']
                
                # Plot the feature
                ax.imshow(projection, cmap='viridis')
                ax.text(0.95, 0.68, f"Ch {channel}\nW:{weight:.3f}\nA:{activation:.3f}", 
                        transform=ax.transAxes, color='white', fontsize=7, ha='right', va='bottom')
                
                ax.set_title("")
                ax.axis('off')
            
            # Plot moving features (right 2 columns)
            for i, feat in enumerate(mov_features):
                # Calculate position: i=0,1 -> row 0, i=2,3 -> row 1
                row = row_start + (i // 2)
                col = 2 + (i % 2)  # 2 or 3
                
                ax = plt.subplot(gs[row, col])
                
                channel = feat['channel']
                activation = feat['activation']
                weight = feat['weight']
                
                # Get XY projection
                projection = feat['projections']['xy']
                
                # Plot the feature
                ax.imshow(projection, cmap='viridis')
                ax.text(0.95, 0.68, f"Ch {channel}\nW:{weight:.3f}\nA:{activation:.3f}", 
                            transform=ax.transAxes,
                            color='white', fontsize=7, ha='right', va='bottom')
                    
                ax.set_title("") 
                ax.axis('off')
        
        # Adjust spacing to make grid more compact
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.04, left=0.05, right=0.97)
        pdf.savefig()
        plt.close()

    def generate_training_dynamics_report(self):
        """Generate PDF report for training dynamics analysis with feature visualizations."""
        # Set specific PDF path for training dynamics
        self.pdf_path = self.output_path / f"{self.base_filename}_training_dynamics.pdf"        
        # Check if we have input and reference paths for feature visualization
        has_feature_data = hasattr(self.settings, 'input_path') and self.settings.input_path and \
                        hasattr(self.settings, 'reference_file') and self.settings.reference_file
        
        feature_viz = None
        if has_feature_data:
            # Find a single file to use for feature visualization
            input_path = Path(self.settings.input_path)
            ref_path = Path(self.settings.reference_file)
            
            # Try to find any TIFF files in the directory
            file_sorter = FileSorting(self.settings)
            organized_files = file_sorter.organize_files()

            sample_files = []
            if organized_files:
                # Get first sample
                first_sample_id = next(iter(organized_files))
                # Get first timepoint of that sample
                first_timepoint_id = next(iter(organized_files[first_sample_id]))
                # Get files from reference channel
                if self.settings.reference_channel in organized_files[first_sample_id][first_timepoint_id]:
                    reference_files = organized_files[first_sample_id][first_timepoint_id][self.settings.reference_channel]
                    if reference_files:
                        sample_files = [reference_files[0]]
                        self.progress_signal.emit(f"Using reference channel {self.settings.reference_channel} file for feature visualization")

            if sample_files and ref_path.exists():              
                # Extract feature visualizations
                feature_viz = self.visualize_features(ref_path, sample_files[0])
                
            else:
                self.progress_signal.emit("No suitable files found for feature visualization")
        
        with PdfPages(self.pdf_path) as pdf:
            # Page 1: Training information and metrics
            self.generate_training_page(pdf)
            
            # Page 2: Feature representations from model levels (if available)
            if feature_viz and len(feature_viz) > 0:
                self.generate_feature_visualization_page(pdf, feature_viz)
            else:
                self.progress_signal.emit("Skipping feature visualization page - no data available")
        
        return self.pdf_path
    
    def generate_evaluation_report(self):
        """Generate PDF report for model evaluation on registration tasks."""
        self.progress_signal.emit(f"Generating evaluation report...")
        self.pdf_path = self.output_path / f"{self.base_filename}_evaluation.pdf"     
        # Find test file pairs based on processing mode
        test_file_pairs = self.find_test_files()
        
        if not test_file_pairs:
            self.progress_signal.emit("No suitable test files found. Cannot generate evaluation report.")
            return None
        
        # Process files and collect visualization data
        reg_viz_batch = self.visualize_registration_batch(test_file_pairs)
        
        with PdfPages(self.pdf_path) as pdf:
            # Generate only the registration results page
            if reg_viz_batch and len(reg_viz_batch) > 0:
                try:
                    self.generate_registration_results_page(pdf, reg_viz_batch)
                    self.progress_signal.emit("Registration results page generated successfully")
                except Exception as e:
                    self.progress_signal.emit(f"Error generating registration results page: {str(e)}")
                    self.progress_signal.emit(traceback.format_exc())
            else:
                self.progress_signal.emit("No registration results to display.")
        
        return self.pdf_path
    
    def _run_training_dynamics(self):
        """Run training dynamics analysis and generate PDF report."""
        self.progress_signal.emit("Generating training dynamics report...")
        
        # Generate specialized training dynamics report
        self.generate_training_dynamics_report()

    def _run_evaluation(self):
        """Run model evaluation on a dataset and generate report."""
        self.progress_signal.emit("Evaluating model performance...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            device_info = torch.cuda.get_device_name(0)
            self.progress_signal.emit(f"Using GPU: {device_info}")
        else:
            self.progress_signal.emit("Using CPU (no CUDA device available)")
        
        # Generate evaluation report
        self.generate_evaluation_report()
        
    def stop(self):
        """Request the worker to stop execution."""
        self.stop_requested = True
        self.progress_signal.emit("Analysis stop requested...")
