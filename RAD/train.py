import os
import threading
import SimpleITK as sitk
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.optim as optim
import torch.cuda.amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tifffile
import numpy as np
import datetime
import time
from engine import RegistrationWorker
from engine import get_metadata
from settings import VisualizationData
from file_sorting import FileSorting
from registration_net import RegistrationNet, TransformLoss
from torch.optim import swa_utils

# Automatically detect the number of available CPU cores
n_threads = os.cpu_count()
# Set the number of threads globally
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)

class TrainingWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    visualization_needed_signal = pyqtSignal(object)
    
    def __init__(self, settings, app):
        super().__init__()
        self.settings = settings
        self.app = app
        self.sort = FileSorting(settings).organize_files()

        # Add these lines after the existing initialization section
        self.visualization_event = threading.Event()
        self.visualization_result = None
        self.stop_requested = False
        self.corrections = {}

        # Initialize training tracking variables
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.final_loss = float('inf')
        self.training_metrics = {}
        
        # Add this new code for enhanced training metrics
        self.loss_trajectories = {}  # Stores loss values by cycle and epoch
        self.lr_history = {}         # Stores learning rate history by cycle
        self.lr_reduction_epochs = {}  # Tracks when learning rate was reduced
        self.epoch_times = {}        # Tracks time per epoch
  
    def run(self):
        try:
            training_start_time = time.time()
            self.visualization_event = threading.Event()
            self.visualization_result = None
            self.stop_requested = False
            self.corrections = {}

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.progress_signal.emit(f"Using device: {device}")
            
            output_dir = Path(self.settings.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize or load model based on mode
            if not self.settings.train_from_scratch and self.settings.model_path:
                model = RegistrationNet().to(device)
                self.progress_signal.emit(f"Loading model from {self.settings.model_path}")
                checkpoint = torch.load(self.settings.model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                self.progress_signal.emit("Model loaded successfully")
            else:
                self.progress_signal.emit("Initializing new model")
                model = RegistrationNet().to(device)

            optimizer = optim.Adam(model.parameters(), lr=self.settings.learning_rate)
            
            # Phase 1: Collect corrections
            if self.settings.is_unsupervised:
                self.progress_signal.emit("\nPreparing unsupervised training data...")
                self.prepare_unsupervised_data()
            else:
                self.progress_signal.emit("\nCollecting corrective transforms...")
                self.collect_corrections(model, device)
            
            # Phase 1.5: Data augmentation (if enabled)
            if hasattr(self.settings, 'augment_data') and self.settings.augment_data:
                self.progress_signal.emit("\nPerforming data augmentation...")
                self.augment_dataset(model, device)
                    
            cycle = 0
            base_target_loss = self.settings.target_loss
            
            while not self.stop_requested:
                self.progress_signal.emit(f"\nStarting cycle {cycle + 1}")
                
                # Decrease target loss with each cycle
                current_target_loss = base_target_loss / (cycle + 1)
                
                # Phase 2: Continue training same model
                model = self.train_model(
                    device,cycle, model=model,optimizer=optimizer,
                    target_loss=current_target_loss, backward_batch_size=self.settings.backward_batch_size)
                
                # Phase 3: Validate predictions
                all_validated = self.validate_predictions(model, device)
                if all_validated:
                    self.progress_signal.emit("\nAll pairs validated successfully!")
                    
                    # Calculate total training time
                    training_total_time = time.time() - training_start_time
                    
                    # Comprehensive metadata for final model
                    final_metadata = {
                        # Training statistics
                        'total_epochs': self.current_epoch,
                        'best_loss': self.best_loss,
                        'final_loss': self.final_loss,
                        'cycles_completed': cycle + 1,
                        'training_time_seconds': training_total_time,
                        'per_cycle_metrics': self.training_metrics,
                        
                        # Training configuration
                        'processing_mode': 'unsupervised' if self.settings.is_unsupervised else self.settings.p_mode,
                        'learning_rate': self.settings.learning_rate,
                        'trained_from_scratch': getattr(self.settings, 'train_from_scratch', True),
                        'reference_channel': self.settings.reference_channel,
                        
                        # Dataset information
                        'n_original_pairs': len([idx for idx in self.corrections.keys() 
                                            if not hasattr(self, 'augmented_pairs') or idx not in self.augmented_pairs]),
                        'n_augmented_pairs': len(self.augmented_pairs) if hasattr(self, 'augmented_pairs') else 0,
                        'augmentation_factor': getattr(self.settings, 'augmentation_factor', 0),
                        
                        # Validation information
                        'all_pairs_validated': all_validated,
                        
                        # System information
                        'device': str(device),
                        'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                        'timestamp': datetime.datetime.now().isoformat(),
                        
                        # Add training dynamics
                        'training_dynamics': {
                            'loss_trajectories': self.loss_trajectories,
                            'lr_reduction_epochs': self.lr_reduction_epochs,
                            'avg_epoch_times': {cycle: np.mean(times) for cycle, times in self.epoch_times.items()},
                            'initial_lr': self.settings.learning_rate,
                            'lr_histories': self.lr_history,
                        }}
                    
                    save_path = output_dir / f'final_model_cycle_{cycle+1}.pth'
                    
                    # Save with standard format
                    torch.save({
                        'state_dict': model.state_dict(),
                        'metadata': final_metadata
                    }, save_path)
                    
                    self.progress_signal.emit(f"Model saved to: {save_path}")
                    break
                    
                cycle += 1
                
            # Clean up temporary files created during augmentation
            if hasattr(self, 'augmented_pairs'):
                self.progress_signal.emit("\nCleaning up temporary augmentation files...")
                temp_dir = Path(self.settings.output_path) / "temp_augmentation"
                if temp_dir.exists():
                    for temp_file in temp_dir.glob('*.tif'):
                        try:
                            temp_file.unlink()
                        except Exception as e:
                            self.progress_signal.emit(f"Warning: Could not delete {temp_file.name}: {str(e)}")
                    try:
                        temp_dir.rmdir()
                    except Exception as e:
                        self.progress_signal.emit(f"Warning: Could not remove temp directory: {str(e)}")
                
        except Exception as e:
            self.progress_signal.emit(f"Error during training: {str(e)}")
        finally:
            self.finished_signal.emit()

    def collect_corrections(self, model, device):
        """Collect initial corrections using processing mode logic"""
        pair_idx = 0
        organized_files = self.sort

        # Get total number of pairs based on mode
        total_pairs = self._count_total_pairs(organized_files)
        # Limit pairs based on batch size
        pairs_to_process = min(total_pairs, self.settings.batch_size)
        self.progress_signal.emit(f"Processing {pairs_to_process} pairs out of {total_pairs} total pairs")

        if self.settings.p_mode in ['batch', 'global']:
            # Process all samples with reference file
            ref_img = self.load_image_for_model(self.settings.reference_file)
            
            for sample_id, timepoints in organized_files.items():
                if pair_idx >= pairs_to_process:
                    break
                if self.settings.p_mode == 'global':
                    # Handle global reference mode
                    if self.settings.reference_timepoint == 'First':
                        ref_timepoint = min(timepoints.keys())
                    else:  # LAST
                        ref_timepoint = max(timepoints.keys())
                    # Get transform from reference timepoint
                    moving_path = timepoints[ref_timepoint][self.settings.reference_channel][0]
                else:
                    # Handle batch mode
                    timepoint_id = next(iter(timepoints))
                    moving_path = timepoints[timepoint_id][self.settings.reference_channel][0]

                # Process the pair
                moving = self.load_image_for_model(moving_path)
                
                viz_data = VisualizationData(
                    moving=moving.squeeze().squeeze(0),
                    reference=ref_img.squeeze().squeeze(0),
                    predicted_transform=torch.zeros(6),
                    file_pair=(moving_path, self.settings.reference_file),
                    mode='correction'
                )
                
                self.visualization_needed_signal.emit(viz_data)
                self.visualization_event.wait()
                self.visualization_event.clear()
                
                if self.visualization_result is not None:
                    self.corrections[pair_idx] = self.visualization_result.clone()
                pair_idx += 1

        elif self.settings.p_mode == 'channel':
            # Process channel alignment
            self.progress_signal.emit(f"Processing paired alignment in channel mode...")
            
            # Iterate through all samples
            for sample_id, timepoints in organized_files.items():
                if pair_idx >= pairs_to_process:
                    break
                    
                self.progress_signal.emit(f"Processing condition/sample {sample_id}...")
                timepoint_id = next(iter(timepoints))
                channels = timepoints[timepoint_id]
                ref_path = channels[self.settings.reference_channel][0]
                ref_img = self.load_image_for_model(ref_path)

                for channel_id, files in channels.items():
                    if pair_idx >= pairs_to_process:
                        break
                    if channel_id == self.settings.reference_channel:
                        continue
                    
                    self.progress_signal.emit(f"Processing pair: Sample {sample_id}, Channel {channel_id}")
                    moving = self.load_image_for_model(files[0])

                    viz_data = VisualizationData(
                        moving=moving.squeeze().squeeze(0),
                        reference=ref_img.squeeze().squeeze(0),
                        predicted_transform=torch.zeros(6),
                        file_pair=(files[0], ref_path),
                        mode='correction'
                    )
                    
                    self.visualization_needed_signal.emit(viz_data)
                    self.visualization_event.wait()
                    self.visualization_event.clear()
                    
                    if self.visualization_result is not None:
                        self.corrections[pair_idx] = self.visualization_result.clone()
                        self.progress_signal.emit(f"Stored correction for pair {pair_idx}")
                    pair_idx += 1

        else:  # Drift or jitter correction
            if self.settings.p_mode == 'drift':
                # Process sequential timepoints
                sorted_timepoints = sorted(organized_files[1].keys())
                ref_path = organized_files[1][sorted_timepoints[0]][self.settings.reference_channel][0]
                ref_img = self.load_image_for_model(ref_path)

                for t1, t2 in zip(sorted_timepoints[:-1], sorted_timepoints[1:]):
                    if pair_idx >= pairs_to_process:  # Add this check
                        break

                    moving_path = organized_files[1][t2][self.settings.reference_channel][0]
                    moving = self.load_image_for_model(moving_path)

                    viz_data = VisualizationData(
                        moving=moving.squeeze().squeeze(0),
                        reference=ref_img.squeeze().squeeze(0),
                        predicted_transform=torch.zeros(6),
                        file_pair=(moving_path, ref_path),
                        mode='correction'
                    )
                    
                    self.visualization_needed_signal.emit(viz_data)
                    self.visualization_event.wait()
                    self.visualization_event.clear()
                    
                    if self.visualization_result is not None:
                        self.corrections[pair_idx] = self.visualization_result.clone()
                    
                    # Update reference for next iteration
                    ref_img = moving
                    ref_path = moving_path
                    pair_idx += 1

            elif self.settings.p_mode == 'jitter':
                # Create averaged reference
                sorted_timepoints = sorted(organized_files[1].keys())
                half_window = self.settings.window_size // 2
                ref_pos = sorted_timepoints.index(self.settings.ref_timepoint) if self.settings.ref_timepoint in sorted_timepoints else len(sorted_timepoints) // 2
                window_start = max(0, ref_pos - half_window)
                window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)

                # Get reference frames
                ref_frames = []
                for t in range(window_start, window_end):
                    ref_files = organized_files[1][sorted_timepoints[t]][self.settings.reference_channel]
                    ref_frames.extend(ref_files)

                # Create averaged reference
                first_file = ref_frames[0]
                pixel_size_x, pixel_size_y, pixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(first_file)
                img = tifffile.imread(first_file).astype(np.float64)
                sum_array = np.zeros(img.shape, dtype=np.float16)

                for file in ref_frames:
                    sum_array += tifffile.imread(file).astype(np.float16)

                avg_array = sum_array / len(ref_frames)
                if pixel_type == np.uint8:
                    avg_array = ((avg_array - avg_array.min()) / (avg_array.max() - avg_array.min()) * 255).astype(np.uint8)
                else:
                    avg_array = avg_array.astype(np.uint16)

                ref_path = str(Path(self.settings.output_path) / "average_stack.tif")
                tifffile.imwrite(ref_path, avg_array, imagej=True, resolution=(1./pixel_size_x, 1./pixel_size_y),
                    metadata={'spacing': pixel_size_z, 'unit': physical_size_unit, 'axes': dimension_order})
                ref_img = self.load_image_for_model(ref_path)

                # Process all timepoints against averaged reference
                for timepoint_id in sorted_timepoints:
                    if pair_idx >= pairs_to_process:  # Add this check
                        break
                    if window_start <= sorted_timepoints.index(timepoint_id) < window_end:
                        continue
                    
                    moving_path = organized_files[1][timepoint_id][self.settings.reference_channel][0]
                    moving = self.load_image_for_model(moving_path)
                    
                    viz_data = VisualizationData(
                        moving=moving.squeeze().squeeze(0),
                        reference=ref_img.squeeze().squeeze(0),
                        predicted_transform=torch.zeros(6),
                        file_pair=(moving_path, ref_path),
                        mode='correction'
                    )
                    
                    self.visualization_needed_signal.emit(viz_data)
                    self.visualization_event.wait()
                    self.visualization_event.clear()
                    
                    if self.visualization_result is not None:
                        self.corrections[pair_idx] = self.visualization_result.clone()
                    pair_idx += 1

    def _train_null_transformation(self, model, device, loss_fn, optimizer, scaler, organized_files):
        """Train the model on a null transformation (reference to itself)"""
        # Skip null transformation training for channel mode or unsupervised training
        if self.settings.p_mode == 'channel' or hasattr(self.settings, 'is_unsupervised') and self.settings.is_unsupervised:
            return None 
        # Get appropriate reference image based on processing mode
        if self.settings.p_mode in ['batch', 'global']:
            reference = self.load_image_for_model(self.settings.reference_file)
        elif self.settings.p_mode == 'drift':
            sorted_timepoints = sorted(organized_files[1].keys())
            first_timepoint = sorted_timepoints[0]
            ref_path = organized_files[1][first_timepoint][self.settings.reference_channel][0]
            reference = self.load_image_for_model(ref_path)
        else:  # jitter mode
            ref_path = str(Path(self.settings.output_path) / "average_stack.tif")
            reference = self.load_image_for_model(ref_path)

        # Train with reference-to-reference (null transformation)
        loss = None
        try:
            moving = reference.clone()
            reference = reference.to(device)
            moving = moving.to(device)
            target_transform = torch.zeros(6, device=device)
            
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                predicted_transform = model(reference, moving)
                loss = loss_fn(predicted_transform, target_transform)
                loss = loss*5
                      
            self.progress_signal.emit(f"  Null transform Loss: {loss.item():.6f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                self.progress_signal.emit(f"Out of memory on null transformation step")
                return None
            raise e
        finally:
            del reference, moving
            torch.cuda.empty_cache()
        
        return loss

    def prepare_unsupervised_data(self):
        """Prepare data for unsupervised training using reference/moving pairs"""
        pair_idx = 0
        organized_files = self.sort
        
        # Count all available pairs
        total_pairs = 0
        for sample_id, timepoints in organized_files.items():
            for timepoint_id, channels in timepoints.items():
                if 0 in channels and 1 in channels:
                    total_pairs += 1
        
        pairs_to_process = min(total_pairs, self.settings.batch_size)
        self.progress_signal.emit(f"Processing {pairs_to_process} aligned image pairs out of {total_pairs} total pairs")
        
        # In unsupervised mode, we ignore processing mode and just look for:
        # - Channel 0 as the reference image
        # - Channel 1 as the aligned/moving image
        # We use null transforms (zeros) as the target since images are already aligned
        
        for sample_id, timepoints in organized_files.items():
            if pair_idx >= pairs_to_process:
                break
                
            for timepoint_id, channels in timepoints.items():
                if pair_idx >= pairs_to_process:
                    break
                    
                # Check if both channel 0 and 1 exist for this sample/timepoint
                if 0 in channels and 1 in channels:
                    ref_path = channels[0][0]
                    moving_path = channels[1][0]
                    
                    self.progress_signal.emit(f"Processing pair {pair_idx + 1}: Sample {sample_id}, Timepoint {timepoint_id}")
                    self.progress_signal.emit(f"Reference: {ref_path.name}, Moving: {moving_path.name}")
                    
                    # Create null transform (zeros) for this pair - they should already be aligned
                    self.corrections[pair_idx] = torch.zeros(6)
                    pair_idx += 1
                else:
                    self.progress_signal.emit(f"Warning: Sample {sample_id}, Timepoint {timepoint_id} does not have both channel 0 (reference) and channel 1 (moving)")

    def train_model(self, device, cycle, model=None, optimizer=None, target_loss=1e-3, backward_batch_size=1):
        """Train model using collected corrections"""
        self.progress_signal.emit("\nTraining on corrections...")
        self.progress_signal.emit(f"Target loss for cycle {cycle}: {target_loss:.6f}")

        loss_fn = TransformLoss()
        scaler = torch.cuda.amp.GradScaler()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        # Initialize SWA model and scheduler
        swa_model = swa_utils.AveragedModel(model)
        swa_start = 10  # Start SWA after this many epochs
        swa_scheduler = swa_utils.SWALR(
            optimizer, 
            swa_lr=self.settings.learning_rate * 0.1,  # SWA typically uses lower learning rate
            anneal_epochs=5  # Gradually anneal learning rate over these many epochs
        )
        use_swa = True  # Flag to enable/disable SWA
        swa_active = False  # Flag to track if SWA is currently active
        best_loss = float('inf')
        epoch = 0
        organized_files = self.sort
        
        # Track metrics for this cycle
        cycle_metrics = {
            'target_loss': target_loss,
            'start_time': time.time(),
            'best_loss': float('inf'),
            'final_loss': None,
            'epochs': 0,
            'learning_rate_history': []
        }

        # Initialize loss trajectory and learning rate history for this cycle
        self.loss_trajectories[cycle] = []
        self.lr_history[cycle] = []
        self.lr_reduction_epochs[cycle] = []
        self.epoch_times[cycle] = []

        while True:
            epoch_start_time = time.time()
            sample_count = 0 
            total_loss = 0
            model.train()

            # Split corrections into smaller backward batches
            correction_items = list(self.corrections.items())
            backward_batches = [correction_items[i:i+backward_batch_size] 
                            for i in range(0, len(correction_items), backward_batch_size)]

            # Process each backward batch
            for batch_idx, batch_items in enumerate(backward_batches):
                # Zero gradients at the beginning of each backward batch
                optimizer.zero_grad()
                
                batch_loss = 0.0
                batch_samples = 0

                for idx, target_transform in batch_items:
                    try:
                        # Clear memory before loading new data
                        torch.cuda.empty_cache()
                        # Get the corresponding pair based on processing mode
                        if hasattr(self, 'augmented_pairs') and idx in self.augmented_pairs:
                            # For augmented pairs, load from saved paths
                            pair_info = self.augmented_pairs[idx]
                            moving = self.load_image_for_model(pair_info['moving_path']).to(device)
                            reference = self.load_image_for_model(pair_info['reference_path']).to(device)

                        elif hasattr(self.settings, 'is_unsupervised') and self.settings.is_unsupervised:
                            found_sample_id = None
                            sample_index = 0
                            
                            for sample_id in organized_files:
                                if sample_index == idx:
                                    found_sample_id = sample_id
                                    break
                                sample_index += 1
                            
                            if found_sample_id is None:
                                self.progress_signal.emit(f"Warning: Could not find sample for correction index {idx}")
                                continue
                            
                            # Get the first timepoint for this sample
                            timepoint_id = next(iter(organized_files[found_sample_id].keys()))
                            channels = organized_files[found_sample_id][timepoint_id]
                            
                            # In unsupervised mode, channel 0 is always reference, channel 1 is always moving
                            if 0 in channels and 1 in channels:
                                ref_path = channels[0][0]
                                moving_path = channels[1][0]
                                
                                reference = self.load_image_for_model(ref_path).to(device)
                                moving = self.load_image_for_model(moving_path).to(device)
                            else:
                                self.progress_signal.emit(f"Warning: Sample {found_sample_id} does not have both channel 0 and 1")
                                continue

                        # Get the corresponding pair based on processing mode
                        elif self.settings.p_mode in ['batch', 'global']:
                            reference = self.load_image_for_model(self.settings.reference_file)
                            # For batch mode, get first timepoint of each sample
                            sample_id = idx + 1  # Assuming indices match sample IDs
                            timepoints = organized_files[sample_id]
                            if self.settings.p_mode == 'global':
                                # Get reference timepoint for global mode
                                if self.settings.reference_timepoint == 'First':
                                    timepoint_id = min(timepoints.keys())
                                else:  # LAST
                                    timepoint_id = max(timepoints.keys())
                            else:
                                # Batch mode - use first timepoint
                                timepoint_id = next(iter(timepoints))
                            moving_path = timepoints[timepoint_id][self.settings.reference_channel][0]
                            moving = self.load_image_for_model(moving_path)

                        elif self.settings.p_mode == 'channel':
                            # For channel mode, idx corresponds to non-reference channels
                            try:
                                # Find which sample this index refers to
                                found_sample_id = None
                                found_channel_id = None
                                sample_indices_processed = 0
                                
                                for sample_id in organized_files:
                                    timepoint_id = next(iter(organized_files[sample_id].keys()))
                                    channels = organized_files[sample_id][timepoint_id]
                                    channel_numbers = [ch for ch in channels.keys() if ch != self.settings.reference_channel]
                                    
                                    # If this index falls within this sample's range
                                    if idx < sample_indices_processed + len(channel_numbers):
                                        found_sample_id = sample_id
                                        local_idx = idx - sample_indices_processed
                                        found_channel_id = channel_numbers[local_idx]
                                        break
                                    
                                    sample_indices_processed += len(channel_numbers)
                                
                                if found_sample_id is None:
                                    raise ValueError(f"Could not find sample for correction index {idx}")
                                
                                # Now we have the sample_id and channel_id for this index
                                timepoint_id = next(iter(organized_files[found_sample_id].keys()))
                                channels = organized_files[found_sample_id][timepoint_id]
                                
                                # Reference is always from reference channel
                                ref_path = channels[self.settings.reference_channel][0]
                                reference = self.load_image_for_model(ref_path)
                                
                                # Get moving image from corresponding channel
                                moving_path = channels[found_channel_id][0]
                                moving = self.load_image_for_model(moving_path)
                                
                                self.progress_signal.emit(f"Training on Sample {found_sample_id}, Channel {found_channel_id}")
                            except Exception as e:
                                self.progress_signal.emit(f"Error loading channel data for index {idx}: {str(e)}")
                                continue

                        elif self.settings.p_mode == 'drift':
                            # For drift mode, idx corresponds to consecutive timepoint pairs
                            sorted_timepoints = sorted(organized_files[1].keys())
                            t1, t2 = sorted_timepoints[idx], sorted_timepoints[idx + 1]
                            
                            ref_path = organized_files[1][t1][self.settings.reference_channel][0]
                            moving_path = organized_files[1][t2][self.settings.reference_channel][0]
                            
                            reference = self.load_image_for_model(ref_path)
                            moving = self.load_image_for_model(moving_path)

                        else:  # jitter mode
                            # Use the averaged reference
                            ref_path = str(Path(self.settings.output_path) / "average_stack.tif")
                            reference = self.load_image_for_model(ref_path)
                            
                            # Get corresponding timepoint
                            sorted_timepoints = sorted(organized_files[1].keys())
                            half_window = self.settings.window_size // 2
                            ref_pos = sorted_timepoints.index(self.settings.ref_timepoint)
                            window_start = max(0, ref_pos - half_window)
                            window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)
                            
                            # Map idx to timepoint outside window
                            timepoints_to_process = [t for t in sorted_timepoints 
                                                if not (window_start <= sorted_timepoints.index(t) < window_end)]
                            current_timepoint = timepoints_to_process[idx]
                            moving_path = organized_files[1][current_timepoint][self.settings.reference_channel][0]
                            moving = self.load_image_for_model(moving_path)

                        # Forward pass and loss calculation
                        reference = reference.to(device)
                        moving = moving.to(device)
                        target_transform = target_transform.to(device)
                        
                        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                            predicted_transform = model(reference, moving)
                            loss = loss_fn(predicted_transform, target_transform)

                        # Accumulate full loss (not scaled)
                        batch_loss += loss.item()
                        batch_samples += 1
                        
                        # Backward pass for each sample
                        scaler.scale(loss).backward()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            self.progress_signal.emit(f"Out of memory skip of sample")
                            continue
                        raise e
                    finally:
                        del reference, moving
                        torch.cuda.empty_cache()

                # Add null transformation if needed (once per backward batch)
                null_loss = self._train_null_transformation(model, device, loss_fn, optimizer, scaler, organized_files)
                if null_loss is not None:
                    batch_loss += null_loss.item()
                    batch_samples += 1
                
                # Only update optimizer if we processed at least one sample
                if batch_samples > 0:
                    # Step optimizer after each backward batch
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update total loss for reporting
                    total_loss += batch_loss
                    sample_count += batch_samples
                
                # Zero gradients after the step
                optimizer.zero_grad()

            # Calculate average loss for reporting
            avg_loss = total_loss / sample_count if sample_count > 0 else float('inf')
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # Track epoch time
            self.epoch_times[cycle].append(epoch_duration)
            
            self.progress_signal.emit(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}, Time: {epoch_duration:.2f}s")
            
            # Track loss trajectory and learning rate
            self.loss_trajectories[cycle].append(avg_loss)
            
            # Track learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.lr_history[cycle].append(current_lr)
            cycle_metrics['learning_rate_history'].append(current_lr)
            
            # Check if learning rate was reduced
            if len(self.lr_history[cycle]) > 1 and self.lr_history[cycle][-1] < self.lr_history[cycle][-2]:
                self.lr_reduction_epochs[cycle].append(epoch)
                self.progress_signal.emit(f"Learning rate reduced to {current_lr:.8f} at epoch {epoch + 1}")
                        
            # Check if we should activate SWA
            if use_swa and epoch >= swa_start and not swa_active:
                self.progress_signal.emit(f"Activating SWA at epoch {epoch + 1}")
                swa_active = True
                # Reset optimizer's learning rate to a consistent value for SWA
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.settings.learning_rate * 0.5
                self.progress_signal.emit(f"Reset learning rate to {self.settings.learning_rate * 0.5} for SWA")

            # Update scheduler and SWA as needed
            if swa_active:
                # Update SWA model weights
                swa_model.update_parameters(model)
                # Step ONLY SWA scheduler, not the regular scheduler
                swa_scheduler.step()
            else:
                # Use regular scheduler only when SWA is not active
                scheduler.step(avg_loss)

            # Checkpointing
            if avg_loss < best_loss:
                best_loss = avg_loss
                cycle_metrics['best_loss'] = best_loss
                
                # Collect metadata
                metadata = {
                    # Training statistics
                    'epochs_completed': epoch,
                    'best_loss': best_loss,
                    'current_loss': avg_loss,
                    'cycle': cycle,
                    'target_loss': target_loss,
                    'learning_rate': current_lr,
                    
                    # Training configuration
                    'processing_mode': self.settings.p_mode,
                    'initial_learning_rate': self.settings.learning_rate,
                    'trained_from_scratch': getattr(self.settings, 'train_from_scratch', True),
                    'reference_channel': self.settings.reference_channel,
                    
                    # Dataset information
                    'n_pairs': len(self.corrections),
                    'n_augmented_pairs': len(self.augmented_pairs) if hasattr(self, 'augmented_pairs') else 0,
                    'augmentation_factor': getattr(self.settings, 'augmentation_factor', 0),
                    
                    # System information
                    'device': str(device),
                    'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                    'timestamp': datetime.datetime.now().isoformat(),

                    # training dynamics metadata
                    'training_dynamics': {
                        'loss_trajectory': self.loss_trajectories[cycle],
                        'lr_reduction_epochs': self.lr_reduction_epochs[cycle],
                        'avg_epoch_time_seconds': np.mean(self.epoch_times[cycle]) if self.epoch_times[cycle] else 0,
                        'initial_lr': self.lr_history[cycle][0] if self.lr_history[cycle] else self.settings.learning_rate,
                        'lr_history': self.lr_history[cycle],
                    }
                }
                
                # Save with enhanced metadata
                checkpoint_path = str(Path(self.settings.output_path) / f'best_model_cycle_{cycle}.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                    'training_info': {
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                    },
                    'metadata': metadata
                }, checkpoint_path)

            if avg_loss < target_loss:
                self.progress_signal.emit(f"Reached target loss: {avg_loss:.6f}")
                break
            epoch += 1

        # Store final metrics for this cycle
        cycle_metrics['final_loss'] = avg_loss
        cycle_metrics['epochs'] = epoch
        cycle_metrics['end_time'] = time.time()
        cycle_metrics['training_time'] = cycle_metrics['end_time'] - cycle_metrics['start_time']
        
        # Update class-level tracking
        self.current_epoch += epoch
        self.best_loss = min(self.best_loss, best_loss)
        self.final_loss = avg_loss
        self.training_metrics[cycle] = cycle_metrics

        # If SWA was used, switch to SWA model
        if use_swa and swa_active:
            self.progress_signal.emit("Switching to SWA model for final evaluation")
            # The module property contains the actual model
            model = swa_model.module

        return model

    def validate_predictions(self, model, device):
        """Validate model predictions using the same data organization as collection"""
        self.progress_signal.emit("\nValidating predictions...")
        
        model.eval()
        all_validated = True
        organized_files = self.sort
        
        original_pairs = {idx: corr for idx, corr in self.corrections.items() 
                        if not hasattr(self, 'augmented_pairs') or idx not in self.augmented_pairs}
        
        # For unsupervised training, validate only the augmented pairs
        if hasattr(self.settings, 'is_unsupervised') and self.settings.is_unsupervised:
            if hasattr(self, 'augmented_pairs') and self.augmented_pairs:
                self.progress_signal.emit("Validating augmented pairs for unsupervised training...")
                with torch.no_grad():
                    for idx, pair_info in self.augmented_pairs.items():
                        # For augmented pairs, load from saved paths
                        moving_path = pair_info['moving_path']
                        ref_path = pair_info['reference_path']
                        
                        try:
                            moving = self.load_image_for_model(moving_path)
                            reference = self.load_image_for_model(ref_path)
                            
                            # Run the prediction
                            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                                predicted_transform = model(reference.to(device), moving.to(device))
                            
                            # Create visualization data for validation
                            viz_data = VisualizationData(
                                moving=moving.squeeze().squeeze(0),
                                reference=reference.squeeze().squeeze(0),
                                predicted_transform=predicted_transform,
                                file_pair=(moving_path, ref_path),
                                mode='validate'
                            )
                            
                            self.visualization_needed_signal.emit(viz_data)
                            self.visualization_event.wait()
                            self.visualization_event.clear()
                            
                            was_validated = self.visualization_result
                            if was_validated is None:
                                all_validated = False
                                self.progress_signal.emit(f"Augmented pair {idx} needs refinement")
                            else:
                                self.progress_signal.emit(f"Augmented pair {idx} validated successfully")
                                
                        except Exception as e:
                            self.progress_signal.emit(f"Error validating augmented pair {idx}: {str(e)}")
                            all_validated = False
                    
                return all_validated
            else:
                self.progress_signal.emit("No augmented pairs found to validate.")
                return True 


        # Only validate the original pairs
        with torch.no_grad():
            for idx in original_pairs.keys():
                # Get corresponding pair using same logic as train_model
                if self.settings.p_mode in ['batch', 'global']:
                    reference = self.load_image_for_model(self.settings.reference_file)
                    # For batch mode, get first timepoint of each sample
                    sample_id = idx + 1  # Assuming indices match sample IDs
                    timepoints = organized_files[sample_id]
                    if self.settings.p_mode == 'global':
                        # Get reference timepoint for global mode
                        if self.settings.reference_timepoint == 'First':
                            ref_timepoint = min(timepoints.keys())
                        else:  # LAST
                            ref_timepoint = max(timepoints.keys())
                        moving_path = timepoints[ref_timepoint][self.settings.reference_channel][0]
                    else:
                        # Batch mode - use first timepoint
                        timepoint_id = next(iter(timepoints))
                        moving_path = timepoints[timepoint_id][self.settings.reference_channel][0]
                    moving = self.load_image_for_model(moving_path)

                elif self.settings.p_mode == 'channel':
                    # For channel mode, idx corresponds to non-reference channels
                    sample_id = 1  # Usually 1 for channel alignment
                    timepoint_id = next(iter(organized_files[sample_id].keys()))
                    channels = organized_files[sample_id][timepoint_id]
                    
                    # Reference is always from reference channel
                    ref_path = channels[self.settings.reference_channel][0]
                    reference = self.load_image_for_model(ref_path)
                    
                    # Get moving image from corresponding channel
                    channel_numbers = [ch for ch in channels.keys() if ch != self.settings.reference_channel]
                    channel_id = channel_numbers[idx]
                    moving_path = channels[channel_id][0]
                    moving = self.load_image_for_model(moving_path)

                elif self.settings.p_mode == 'drift':
                    # For drift mode, idx corresponds to consecutive timepoint pairs
                    sorted_timepoints = sorted(organized_files[1].keys())
                    t1, t2 = sorted_timepoints[idx], sorted_timepoints[idx + 1]
                    
                    ref_path = organized_files[1][t1][self.settings.reference_channel][0]
                    moving_path = organized_files[1][t2][self.settings.reference_channel][0]
                    
                    reference = self.load_image_for_model(ref_path)
                    moving = self.load_image_for_model(moving_path)

                else:  # jitter mode
                    # Use the averaged reference
                    ref_path = str(Path(self.settings.output_path) / "average_stack.tif")
                    reference = self.load_image_for_model(ref_path)
                    
                    # Get corresponding timepoint
                    sorted_timepoints = sorted(organized_files[1].keys())
                    half_window = self.settings.window_size // 2
                    ref_pos = sorted_timepoints.index(self.settings.ref_timepoint)
                    window_start = max(0, ref_pos - half_window)
                    window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)
                    
                    # Map idx to timepoint outside window
                    timepoints_to_process = [t for t in sorted_timepoints 
                                        if not (window_start <= sorted_timepoints.index(t) < window_end)]
                    current_timepoint = timepoints_to_process[idx]
                    moving_path = organized_files[1][current_timepoint][self.settings.reference_channel][0]
                    moving = self.load_image_for_model(moving_path)

                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predicted_transform = model(reference.to(device), moving.to(device))
                
                viz_data = VisualizationData(
                    moving=moving.squeeze().squeeze(0),
                    reference=reference.squeeze().squeeze(0),
                    predicted_transform=predicted_transform,
                    file_pair=(moving_path, ref_path if 'ref_path' in locals() else self.settings.reference_file),
                    mode='validate'
                )
                
                self.visualization_needed_signal.emit(viz_data)
                self.visualization_event.wait()
                self.visualization_event.clear()
                
                was_validated = self.visualization_result
                if was_validated is None:
                    all_validated = False
                    self.progress_signal.emit(f"Pair {idx} needs refinement")
                else:
                    self.progress_signal.emit(f"Pair {idx} validated successfully")

            return all_validated

    def load_image_for_model(self, file_path: Path) -> torch.Tensor:
        """Load and normalize image for model prediction"""
        # Load image
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
    
    def _count_total_pairs(self, organized_files):
        """Count total number of pairs based on processing mode"""
        if self.settings.p_mode in ['batch', 'global']:
            return len(organized_files)  # One pair per sample
        elif self.settings.p_mode == 'channel':
            # Count non-reference channels across all samples
            total = 0
            for sample_id, timepoints in organized_files.items():
                timepoint_id = next(iter(timepoints))
                channels = timepoints[timepoint_id]
                total += len([ch for ch in channels.keys() if ch != self.settings.reference_channel])
            return total
        elif self.settings.p_mode == 'drift':
            # One less than number of timepoints (consecutive pairs)
            timepoints = organized_files[1].keys()
            return len(timepoints) - 1
        elif self.settings.p_mode == 'jitter':
            # Count timepoints outside reference window
            sorted_timepoints = sorted(organized_files[1].keys())
            half_window = self.settings.window_size // 2
            ref_pos = sorted_timepoints.index(self.settings.ref_timepoint) if self.settings.ref_timepoint in sorted_timepoints else len(sorted_timepoints) // 2
            window_start = max(0, ref_pos - half_window)
            window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)
            return len(sorted_timepoints) - (window_end - window_start)    

    def augment_dataset(self, model, device):
        """Generate augmented versions of the dataset by applying random transformations"""
        if not self.settings.augment_data or self.settings.augmentation_factor < 1:
            return
        
        self.progress_signal.emit("\nAugmenting dataset...")
        
        # Track original corrections count
        original_count = len(self.corrections)
        current_idx = original_count
        augmented_pairs = {}
        
        # Create temporary directory for augmentation images
        temp_dir = Path(self.settings.output_path) / "temp_augmentation"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_engine = RegistrationWorker(self.settings)  # Instantiate to access methods
        
        # Process each original correction
        for idx in range(original_count):
            # Get the corresponding image pair
            moving_img, reference_img, moving_path, ref_path = self._get_image_pair_for_idx(idx)
            original_transform = self.corrections[idx].clone()
            
            # Generate augmented samples for this pair
            for aug_idx in range(self.settings.augmentation_factor):
                # Generate random transform
                random_transform = self._generate_random_transform(
                    max_angle=self.settings.max_rotation_angle,
                    max_translation=self.settings.max_translation_factor,
                    image_shape=moving_img.shape[-3:],  # ZYX dimensions
                    reference_path=ref_path
                )
                
                # Create temporary files for the augmentation process
                temp_augmented = temp_dir / f"temp_augmented_{idx}_{aug_idx}.tif"
                
                # Apply the random transform to create a new "moving" image
                # Use existing apply_transform method from engine.py
                temp_engine.apply_transform(
                    input_path=moving_path,
                    reference_path=ref_path,
                    transform=random_transform,
                    output_path=temp_augmented)
                
                # Calculate the inverse transform - this is the correction we need
                inverse_transform = self._invert_transform_sitk(random_transform, ref_path)
                
                # Add to corrections dictionary
                self.corrections[current_idx] = inverse_transform
                
                # Store the augmented pair information for training
                augmented_pairs[current_idx] = {
                    'moving_path': temp_augmented,
                    'reference_path': ref_path
                }
                
                current_idx += 1
                
                # Progress update
                if aug_idx % 5 == 0:
                    self.progress_signal.emit(f"  Generated augmentations for pair {idx+1}/{original_count}")
        
        # Store the augmented pairs for use in training
        self.augmented_pairs = augmented_pairs
        self.progress_signal.emit(f"Dataset augmented: {len(self.corrections)} total pairs "
                                f"({original_count} original + {len(self.corrections) - original_count} augmented)")

    def _generate_random_transform(self, max_angle, max_translation, image_shape, reference_path):
        # Convert max rotation angle from degrees to radians
        max_angle = np.deg2rad(max_angle)  

        # Generate random rotation angles (in radians)
        angle_z = np.random.uniform(-max_angle, max_angle)
        angle_y = np.random.uniform(-max_angle, max_angle)
        angle_x = np.random.uniform(-max_angle, max_angle)

        # Get physical pixel spacing from metadata
        pixel_size_x, pixel_size_y, pixel_size_z, _, _, _, _, _, _ = get_metadata(reference_path)

        # Compute max translation by applying ratio to image dimensions
        z_dim, y_dim, x_dim = image_shape
        max_translation_x = max_translation * x_dim * pixel_size_x
        max_translation_y = max_translation * y_dim * pixel_size_y
        max_translation_z = max_translation * z_dim * pixel_size_z

        # Generate random translations (in physical space)
        trans_x = np.random.uniform(-max_translation_x, max_translation_x)
        trans_y = np.random.uniform(-max_translation_y, max_translation_y)
        trans_z = np.random.uniform(-max_translation_z, max_translation_z)

        # Create transform in the expected order
        transform = torch.tensor([angle_x, angle_y, angle_z, trans_x, trans_y, trans_z], dtype=torch.float32)

        return transform

    def _invert_transform_sitk(self, transform, reference_path):
        """Calculate the mathematically correct inverse of a transform"""        
        # Get image metadata for center calculation
        pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, _, _, _ = get_metadata(reference_path)
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(transform):
            transform_params = transform.detach().cpu().numpy()
            if transform_params.ndim > 1:
                transform_params = transform_params[0]
        else:
            transform_params = np.array(transform)
        
        # Create Euler3D transform
        sitk_transform = sitk.Euler3DTransform()
        sitk_transform.SetParameters(transform_params.tolist())
        
        # Set transform center to image center in physical coordinates
        center = [(size_x - 1)*pixel_size_x/2.0,
                (size_y - 1)*pixel_size_y/2.0,
                (size_z - 1)*pixel_size_z/2.0]
        sitk_transform.SetCenter(center)
        
        # Calculate the inverse transform
        inverse_transform = sitk_transform.GetInverse()
        
        # Get the parameters of the inverse transform
        inverse_params = list(inverse_transform.GetParameters())
        
        # Return as tensor to match the rest of the codebase
        return torch.tensor(inverse_params, dtype=torch.float32)

    def _get_image_pair_for_idx(self, idx):
        """Get the moving and reference images for a given correction index"""
        organized_files = self.sort

        if hasattr(self.settings, 'is_unsupervised') and self.settings.is_unsupervised:
            # In unsupervised mode, channel 0 is reference and channel 1 is moving
            sample_id = idx + 1  # If sample IDs start from 1
            # Get the first timepoint
            timepoint_id = next(iter(organized_files[sample_id].keys()))
            # Get the channels
            channels = organized_files[sample_id][timepoint_id]
            # Get reference (channel 0) and moving (channel 1)
            ref_path = channels[0][0]
            moving_path = channels[1][0]
            reference = self.load_image_for_model(ref_path)
            moving = self.load_image_for_model(moving_path)
            return moving, reference, moving_path, ref_path

        elif self.settings.p_mode in ['batch', 'global']:
            reference = self.load_image_for_model(self.settings.reference_file)
            ref_path = self.settings.reference_file
            
            # For batch/global mode, get appropriate sample
            sample_id = idx + 1  # Assuming indices match sample IDs
            timepoints = organized_files[sample_id]
            
            if self.settings.p_mode == 'global':
                # Reference timepoint for global mode
                if self.settings.reference_timepoint == 'First':
                    timepoint_id = min(timepoints.keys())
                else:  # LAST
                    timepoint_id = max(timepoints.keys())
            else:
                # First timepoint for batch mode
                timepoint_id = next(iter(timepoints))
                
            moving_path = timepoints[timepoint_id][self.settings.reference_channel][0]
            moving = self.load_image_for_model(moving_path)
            
        elif self.settings.p_mode == 'channel':
            # Channel alignment mode
            sample_id = 1  # Usually 1 for channel alignment
            timepoint_id = next(iter(organized_files[sample_id].keys()))
            channels = organized_files[sample_id][timepoint_id]
            
            # Get reference channel
            ref_path = channels[self.settings.reference_channel][0]
            reference = self.load_image_for_model(ref_path)
            
            # Get moving channel
            channel_numbers = [ch for ch in channels.keys() if ch != self.settings.reference_channel]
            channel_id = channel_numbers[idx % len(channel_numbers)]  
            moving_path = channels[channel_id][0]
            moving = self.load_image_for_model(moving_path)
            
        elif self.settings.p_mode == 'drift':
            # Drift mode - consecutive timepoints
            sorted_timepoints = sorted(organized_files[1].keys())
            t1, t2 = sorted_timepoints[idx], sorted_timepoints[idx + 1]
            
            ref_path = organized_files[1][t1][self.settings.reference_channel][0]
            moving_path = organized_files[1][t2][self.settings.reference_channel][0]
            
            reference = self.load_image_for_model(ref_path)
            moving = self.load_image_for_model(moving_path)
            
        else:  # jitter mode
            # Use averaged reference
            ref_path = str(Path(self.settings.output_path) / "average_stack.tif")
            reference = self.load_image_for_model(ref_path)
            
            # Get timepoints outside reference window
            sorted_timepoints = sorted(organized_files[1].keys())
            half_window = self.settings.window_size // 2
            ref_pos = sorted_timepoints.index(self.settings.ref_timepoint)
            window_start = max(0, ref_pos - half_window)
            window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)
            
            timepoints_to_process = [t for t in sorted_timepoints 
                                if not (window_start <= sorted_timepoints.index(t) < window_end)]
            current_timepoint = timepoints_to_process[idx % len(timepoints_to_process)]
            moving_path = organized_files[1][current_timepoint][self.settings.reference_channel][0]
            moving = self.load_image_for_model(moving_path)
        
        return moving, reference, moving_path, ref_path

    def stop(self):
        """Stop the training process"""
        self.stop_requested = True
        self.progress_signal.emit("\nStopping training ...")
