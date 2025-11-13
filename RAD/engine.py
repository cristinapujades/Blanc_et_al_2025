import threading
from pathlib import Path
import torch
from PyQt5.QtCore import QThread, pyqtSignal
import tifffile
import numpy as np
import torch
import SimpleITK as sitk
from tifffile import TiffFile
from typing import  List
from registration_net import RegistrationNet
from settings import RegistrationSettings,VisualizationData
from file_sorting import FileSorting

def get_metadata(file_path):
    tif = TiffFile(file_path)
    imagej = tif.imagej_metadata
    ome = tif.ome_metadata
    # Image Array
    data = tif.asarray()
    # Get the type
    pixel_type=data.dtype
    # Get the array shape
    dimension= data.shape
    # Get the dimension order
    dimension_order = tif.series[0].axes
    # Map dimension names to their sizes
    dimension_sizes = {}
    for i in range(len(dimension)):
        dimension_sizes[dimension_order[i]] = dimension[i]
    size_x=dimension_sizes.get('X', 'N/A')
    size_y=dimension_sizes.get('Y', 'N/A')
    size_z=dimension_sizes.get('Z', 'N/A')
    # Get the pixel size
    x = tif.pages[0].tags['XResolution'].value
    pixel_size_x = x[1] / x[0]
    y = tif.pages[0].tags['YResolution'].value
    pixel_size_y = y[1] / y[0]
    pixel_size_z = imagej.get('spacing', 1.0)
    # Get the physical size unit
    physical_size_unit = imagej['unit']
    return pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order

class RegistrationWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    visualization_needed_signal = pyqtSignal(object)

    def __init__(self, settings: RegistrationSettings):
        super().__init__()
        self.settings = settings
        
        # Initialize additional state variables
        self.visualization_event = threading.Event()
        self.visualization_result = None
        self.stop_requested = False
        self.sort=FileSorting(settings).organize_files()
    
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

    def apply_transform(self, input_path: Path, reference_path: Path, transform: torch.Tensor, output_path: Path):
        """Apply transform while preserving original pixel values and metadata"""
        # Convert any string paths to Path objects at start of function
        input_path = Path(input_path)
        reference_path = Path(reference_path) 
        output_path = Path(output_path)
        
        # Get metadata using the existing function
        Fpixel_size_x, Fpixel_size_y, Fpixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(reference_path)
        fixed = sitk.ReadImage(str(reference_path),sitk.sitkFloat32)  # Reference image
        fixed.SetSpacing((Fpixel_size_x, Fpixel_size_y, Fpixel_size_z))
        pixel_size_x, pixel_size_y, pixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(input_path)
        moving = sitk.ReadImage(str(input_path),sitk.sitkFloat32)   # Image to be registered
        moving.SetSpacing((pixel_size_x, pixel_size_y, pixel_size_z))
        # 1. Identity transform for spatial normalization
        null_transform = sitk.Euler3DTransform()
        identity_transformed = sitk.Resample(moving, fixed, null_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
 
        # 2. Convert prediction into SITK transform
        # Convert tensor to numpy if needed
        if torch.is_tensor(transform):
                transform_params = transform.detach().cpu().numpy()
                if transform_params.ndim > 1:
                    transform_params = transform_params[0]
        else:
                transform_params = transform.copy()

        current_transform = sitk.Euler3DTransform()
        current_transform.SetParameters(transform_params.tolist())

        # Set transform center
        size = fixed.GetSize()
        center = [(size[0] - 1)*Fpixel_size_x/ 2.0,
                  (size[1] - 1)*Fpixel_size_y/ 2.0,
                  (size[2] - 1)*Fpixel_size_z/ 2.0]
        current_transform.SetCenter(center)

        # 3. Apply predicted transform to normalized image
        transformed = sitk.Resample(identity_transformed, fixed,
                current_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        # Convert to array maintaining original dtype
        array= sitk.GetArrayFromImage(transformed).astype(np.uint16)
        if pixel_type == np.uint8:
            array_min = array.min()
            array_max = array.max()
            if array_max > array_min:
                a = ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
            else:
                a = array.astype(np.uint8)
        else:
            a = array
    
        # Save with metadata from get_metadata function
        tifffile.imwrite(str(output_path), a, imagej=True, resolution=(1./Fpixel_size_x, 1./Fpixel_size_y),
            metadata={'spacing': Fpixel_size_z, 'unit': physical_size_unit, 'axes': dimension_order})

    def apply_composite_transform(self, input_path: Path, reference_path: Path, transforms: list, output_path: Path):
        # Convert any string paths to Path objects
        input_path = Path(input_path)
        reference_path = Path(reference_path) 
        output_path = Path(output_path)
        
        # Get metadata using the existing function
        Fpixel_size_x, Fpixel_size_y, Fpixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(reference_path)
        fixed = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)  # Reference image
        fixed.SetSpacing((Fpixel_size_x, Fpixel_size_y, Fpixel_size_z))
        pixel_size_x, pixel_size_y, pixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(input_path)
        moving = sitk.ReadImage(str(input_path), sitk.sitkFloat32)   # Image to be registered
        moving.SetSpacing((pixel_size_x, pixel_size_y, pixel_size_z))
        
        # 1. Identity transform for spatial normalization
        null_transform = sitk.Euler3DTransform()
        identity_transformed = sitk.Resample(moving, fixed, null_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    
        # 2. Create a composite transform from the list of transforms
        composite_transform = sitk.CompositeTransform(3)  # 3D transform
        
        # Calculate transform center once for consistency
        size = fixed.GetSize()
        center = [(size[0] - 1)*Fpixel_size_x/2.0,
                (size[1] - 1)*Fpixel_size_y/2.0,
                (size[2] - 1)*Fpixel_size_z/2.0]
        
        # Add each transform to the composite transform
        for transform in transforms:
            if isinstance(transform, sitk.Transform):
                # If it's already a SITK transform, add it directly
                composite_transform.AddTransform(transform)
            else:
                # Otherwise, convert parameters to a transform
                if torch.is_tensor(transform):
                    transform_params = transform.detach().cpu().numpy()
                    if transform_params.ndim > 1:
                        transform_params = transform_params[0]
                else:
                    transform_params = np.array(transform) if not isinstance(transform, np.ndarray) else transform.copy()
                
                # Create a new transform from the parameters
                current_transform = sitk.Euler3DTransform()
                current_transform.SetParameters(transform_params.tolist())
                current_transform.SetCenter(center)
                
                # Add to composite transform
                composite_transform.AddTransform(current_transform)
        
        # 3. Apply composite transform to normalized image
        transformed = sitk.Resample(identity_transformed, fixed,
                                composite_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        
        # Convert to array maintaining original dtype
        array = sitk.GetArrayFromImage(transformed).astype(np.uint16)
        if pixel_type == np.uint8:
            array_min = array.min()
            array_max = array.max()
            if array_max > array_min:
                a = ((array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
            else:
                a = array.astype(np.uint8)
        else:
            a = array
        
        # Save with metadata from get_metadata function
        tifffile.imwrite(str(output_path), a, imagej=True, resolution=(1./Fpixel_size_x, 1./Fpixel_size_y),
            metadata={'spacing': Fpixel_size_z, 'unit': physical_size_unit, 'axes': dimension_order})

    def process_channels(self, group_channels: List[Path], output_path: Path, 
                        transform: torch.Tensor, reference_path: Path):
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply transform to all files in the group
        for file_path in group_channels:
            output_name = file_path.name  # Preserve original filename pattern
            self.apply_transform(file_path, reference_path, transform, output_path / output_name)

    def process_tensor_for_visualization(self, tensor):
        """Ensure tensor is in [D, H, W] format for visualization"""
        if tensor is None:
            return None
        return tensor.squeeze()  # Remove singleton dimensions to get [D, H, W]

    def run(self):
        try:
            output_path = Path(self.settings.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get organized files
            organized_files = self.sort

            if not organized_files:
                self.progress_signal.emit("No files found to process")
                return

            # Process based on mode
            if self.settings.p_mode in ['channel', 't_channel']:
                self.process_channel_alignment(organized_files)
            elif self.settings.p_mode in ['batch', 'global']:
                self.process_standard_alignment(organized_files)
            else:
                self.process_time_alignment(organized_files)
        except Exception as e:
            self.progress_signal.emit(f"Error during processing: {str(e)}")
        finally:
            self.finished_signal.emit()

    def process_channel_alignment(self, organized_files):
        for sample_id, timepoints in organized_files.items():

                    timepoint_id = next(iter(timepoints))
                    channels = timepoints[timepoint_id]
                    # Get reference channel (fixed image)
                    ref_channel_path = Path(channels[self.settings.reference_channel][0])
                    ref_img = self.process_tensor_for_visualization(self.load_image_for_model(ref_channel_path))
                    # Copy reference channel directly
                    output_path = Path(self.settings.output_path)
                    # Create output dir if needed
                    output_path.mkdir(parents=True, exist_ok=True)
                    outpath = output_path / ref_channel_path.name
                    outpath.write_bytes(ref_channel_path.read_bytes())
                            
                    # Process other channels
                    for channel_id in channels.keys():
                        if channel_id == self.settings.reference_channel:
                                    continue
                        else:       
                            # Compute transform
                            if self.settings.model_path is not None:
                                transform = self.find_transform(channels[channel_id][0], ref_channel_path)
                            else:
                                transform = self.get_transform(channels[channel_id][0], ref_channel_path,ref_img)
                                if torch.is_tensor(transform):
                                        transform = transform.detach().cpu().numpy()
                                if transform.ndim > 1:
                                            transform = transform[0]      
                            # Apply transform to moving channel
                            for timepoint_id, channels in timepoints.items():
                                self.apply_transform(channels[channel_id][0], ref_channel_path, transform, self.settings.output_path/ Path(channels[channel_id][0]).name)
                                self.progress_signal.emit(f"Sample{sample_id} Timepoint{timepoint_id} Channel{channel_id} aligned")

    def process_standard_alignment(self, organized_files):
        """Process alignment for non-channel alignment modes"""
        # make ref image
        ref_img = self.process_tensor_for_visualization(self.load_image_for_model(self.settings.reference_file))
        for sample_id, timepoints in organized_files.items():
            if self.settings.p_mode == 'global':
                # Process all timepoints with the same transform
                if self.settings.reference_timepoint == 'First':
                    ref_timepoint = min(timepoints.keys())
                else:  # LAST
                    ref_timepoint = max(timepoints.keys())
                # Get transform from reference timepoint
                ref_channel_path = timepoints[ref_timepoint][self.settings.reference_channel][0]

                if self.settings.model_path is not None:
                    transform = self.find_transform(ref_channel_path, self.settings.reference_file)
                else:
                    transform = self.get_transform(ref_channel_path, self.settings.reference_file, ref_img)
                # Apply to all timepoints and channels
                for timepoint_id, channels in timepoints.items():
                        for channel_id in channels.keys():
                                self.apply_transform(channels[channel_id][0],  self.settings.reference_file, transform, self.settings.output_path/ Path(channels[channel_id][0]).name)
                                self.progress_signal.emit(f"Sample{sample_id} Timepoint{timepoint_id} Channel{channel_id} aligned")

            else:  # batch without time
                for timepoint_id, channels in timepoints.items():
                    moving_channel_path = channels[self.settings.reference_channel][0]
                    if self.settings.model_path is not None:
                        transform = self.find_transform(moving_channel_path,self.settings.reference_file)
                    else:
                        transform = self.get_transform(moving_channel_path,self.settings.reference_file, ref_img)
                    if transform is not None:
                        # Apply to all channels
                        for channel_id, file_paths in channels.items():
                            for file_path in file_paths:
                                self.process_channels([file_path], self.settings.output_path,
                                    transform=transform, reference_path=self.settings.reference_file)
                                
    def process_time_alignment(self, organized_files):
        for sample_id, timepoints in organized_files.items():
            # Get sorted timepoint keys to ensure sequential processing
            timepoint_ids = sorted(timepoints.keys())
            
            if self.settings.p_mode == "drift":
                # Copy first timepoint directly
                first_timepoint = timepoint_ids[0]
                first_channels = timepoints[first_timepoint]
                
                # Copy all files from first timepoint to output
                for channel_id, file_paths in first_channels.items():
                    for file_path in file_paths:
                        file_path = file_path if isinstance(file_path, Path) else Path(file_path)
                        outpath = self.settings.output_path / file_path.name
                        outpath.write_bytes(file_path.read_bytes())
                
                # Initialize reference image and path
                ref_path = first_channels[self.settings.reference_channel][0]
                ref_path = ref_path if isinstance(ref_path, Path) else Path(ref_path)
                ref_img = self.process_tensor_for_visualization(
                    self.load_image_for_model(ref_path))
                
                # Initialize an empty list to store transform parameters
                transform_history = []
    
                # Process remaining timepoints sequentially
                for timepoint_id in timepoint_ids[1:]:
                    channels = timepoints[timepoint_id]
                    
                    # Get moving channel path
                    moving_channel_path = channels[self.settings.reference_channel][0]
                    moving_channel_path = moving_channel_path if isinstance(moving_channel_path, Path) else Path(moving_channel_path)
                    
                    # First apply existing transforms to get pre-aligned image
                    residual_transform = None
                    if transform_history:
                        # Create a temporary transformed version of the moving image
                        temp_output = self.settings.output_path / f"temp_aligned_{Path(moving_channel_path).name}"
                        self.apply_composite_transform(moving_channel_path, ref_path, transform_history, temp_output)
                        
                        # Find the residual transform using the pre-aligned image
                        if self.settings.model_path is not None:
                            residual_transform = self.find_transform(temp_output, ref_path)
                        else:
                            residual_transform = self.get_transform(temp_output, ref_path, ref_img)
                            
                        # Clean up temporary file
                        if temp_output.exists():
                            temp_output.unlink()
                    else:
                        # First timepoint after reference, just get the direct transform
                        if self.settings.model_path is not None:
                            residual_transform = self.find_transform(moving_channel_path, ref_path)
                        else:
                            residual_transform = self.get_transform(moving_channel_path, ref_path, ref_img)
                    
                    self.progress_signal.emit(f'Residual Transform: {residual_transform}')
                    
                    if residual_transform is not None:
                        # Add the residual transform to history
                        transform_history.append(residual_transform)
                        
                        # Apply all transforms in history to each channel
                        for channel_id, file_paths in channels.items():
                            for file_path in file_paths:
                                output_file = self.settings.output_path / Path(file_path).name
                                self.apply_composite_transform(file_path, ref_path, 
                                                            transform_history, output_file)
                                self.progress_signal.emit(f"Sample {sample_id} Timepoint {timepoint_id} Channel {channel_id} aligned")

                        # Update reference for next timepoint
                        new_ref_path = self.settings.output_path / moving_channel_path.name
                        ref_img = self.process_tensor_for_visualization(
                            self.load_image_for_model(new_ref_path))
                        ref_path = new_ref_path
 
            else:  #jitter

                sorted_timepoints = sorted(organized_files[1].keys())

                # Calculate reference window
                half_window = self.settings.window_size // 2
                ref_pos = sorted_timepoints.index(self.settings.ref_timepoint) if self.settings.ref_timepoint in sorted_timepoints else len(sorted_timepoints) // 2
                window_start = max(0, ref_pos - half_window)
                window_end = min(len(sorted_timepoints), ref_pos + half_window + 1)

                # Get reference frames
                ref_frames = []
                for t in range(window_start, window_end):
                    ref_files = next(files for channel, files in organized_files[1][sorted_timepoints[t]].items() 
                                    if channel == self.settings.reference_channel)
                    ref_frames.extend(ref_files)

                # Read first image to get metadata and shape
                pixel_size_x, pixel_size_y, pixel_size_z, _, _, _, pixel_type, physical_size_unit, dimension_order = get_metadata(ref_frames[0])
                img=tifffile.imread(ref_frames[0]).astype(np.float64)
                sum_array = np.zeros(img.shape, dtype=np.float16)

                # Read and sum each 3D TIFF stack
                for file in ref_frames:
                    sum_array += tifffile.imread(file).astype(np.float16)     # Read the full 3D stack

                # Compute the average
                avg_array = sum_array / len(ref_frames)

                if pixel_type == np.uint8:
                    array_min = avg_array.min()
                    array_max = avg_array.max()
                    if array_max > array_min:
                        avg_array= ((avg_array - array_min) / (array_max - array_min) * 255).astype(np.uint8)
                    else:
                        avg_array = avg_array.astype(np.uint8)
                else:
                    avg_array = avg_array.astype(np.uint16)

                # Save the averaged stack with metadata
                output_file = "average_stack.tif"
                ref_path=str(self.settings.output_path/output_file)
                tifffile.imwrite(ref_path, avg_array, imagej=True, resolution=(1./pixel_size_x, 1./pixel_size_y),
                    metadata={'spacing': pixel_size_z, 'unit': physical_size_unit, 'axes': dimension_order})
                ref_img = self.process_tensor_for_visualization(self.load_image_for_model(ref_path))

                for timepoint_id, channels in timepoints.items():
                    moving_channel_path = channels[self.settings.reference_channel][0]
                    
                    if self.settings.model_path is not None:
                        transform = self.find_transform(moving_channel_path, ref_path)
                    else:
                        transform = self.get_transform(moving_channel_path, ref_path, ref_img)
           
                    if transform is not None:
                        # Apply to all channels
                        for channel_id in channels.keys():
                                output_file = Path(self.settings.output_path) / (channels[channel_id][0]).name
                                self.apply_transform(channels[channel_id][0], ref_path, transform, output_file)
                                self.progress_signal.emit(f"Sample{sample_id} Timepoint{timepoint_id} Channel{channel_id} aligned")

    def get_transform(self, moving_path, ref_path, ref_img):
        """Get transform through visualization UI"""
        moving_img = self.process_tensor_for_visualization(
            self.load_image_for_model(moving_path))

        viz_data = VisualizationData(
            moving=moving_img, 
            reference=ref_img,
            predicted_transform=torch.zeros(6),
            file_pair=(moving_path, ref_path)
        )
        
        self.visualization_needed_signal.emit(viz_data)
        self.visualization_event.wait()
        self.visualization_event.clear()

        if self.visualization_result is not None:
            transform = self.visualization_result
            if torch.is_tensor(transform):
                transform = transform.detach().cpu().numpy()
                if transform.ndim > 1:
                    transform = transform[0]
                    
            # Convert translation components to physical units
            pixel_size_x, pixel_size_y, pixel_size_z, *_ = get_metadata(ref_path)
            transform[3:] = [
                transform[3] * pixel_size_x,
                transform[4] * pixel_size_y,
                transform[5] * pixel_size_z
            ]
            return transform
        return None

    def find_transform(self, moving_path, ref_path):
        reference_image = self.load_image_for_model(ref_path)
        moving_image = self.load_image_for_model(moving_path)
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RegistrationNet()
        checkpoint = torch.load(self.settings.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        # Compute transform
        with torch.no_grad():
            transform = model(reference_image, moving_image)       
        if torch.is_tensor(transform):
            transform = transform.detach().cpu().numpy()
            if transform.ndim > 1:
                transform = transform[0]
        return transform

    def stop(self):
        """Request worker to stop processing"""
        self.stop_requested = True