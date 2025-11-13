from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import tifffile

@dataclass
class TrainingSettings():
    """Settings specific to model training"""
    input_path: Path
    output_path: Path
    reference_channel: int
    p_mode: str
    model_path: Optional[Path] = None
    learning_rate: float = 0.05
    target_loss: float = 0.0001
    batch_size: int = 4
    backward_batch_size: int = 4
    train_from_scratch: bool = True
    sample_prefix: str = ''
    channel_prefix: str = ''
    time_prefix: str = ''
    reference_file: Optional[Path] = None
    window_size: Optional[int] = None
    ref_timepoint: Optional[int] = None
    reference_timepoint: Optional[str] = None
    is_timelapse: bool = False
    augment_data: bool = False
    augmentation_factor: int = 5  # How many augmented samples per original
    max_rotation_angle: float = 15.0  # Maximum rotation in degrees (1-15)
    max_translation_factor: float = 0.1  # Maximum translation as fraction of dimension (0-0.15)
    is_unsupervised: bool = False  # Whether to use unsupervised training mode

@dataclass
class RegistrationSettings():
    """Settings for supervised alignment"""
    input_path: Path
    output_path: Path
    reference_channel: int
    p_mode: str
    model_path: Optional[Path] = None 
    reference_file: Optional[Path] = None
    window_size: Optional[int] = None
    ref_timepoint: Optional[int] = None
    reference_timepoint: Optional[str]=None
    is_timelapse: bool = False
    sample_prefix: Optional[str]=None
    channel_prefix: Optional[str]=None
    time_prefix: Optional[str]=None

@dataclass
class AnalysisSettings():
    """Settings for model analysis"""
    model_path: Path
    output_path: Path
    p_mode: str 
    reference_channel: int  
    input_path: Optional[Path] = None
    reference_file: Optional[Path] = None  
    analysis_mode: str = 'training_dynamics'
    window_size: Optional[int] = None  # For jitter mode
    ref_timepoint: Optional[int] = None  # For jitter mode
    reference_timepoint: Optional[str] = None  # For global mode
    sample_prefix: Optional[str] = None  # For file sorting
    channel_prefix: Optional[str] = None
    time_prefix: Optional[str] = None
    is_timelapse: bool = False  # For timelapse modes


############### Settings Factory

class SettingsFactory:
    """Factory class to create appropriate settings instance from UI input"""

    def create_settings(ui_settings: dict):
        """Create settings instance based on UI mode"""
        if ui_settings.get('training_radio') or ui_settings.get('autotraining_radio'):
            return SettingsFactory.create_training_settings(ui_settings)
        elif ui_settings.get('training_dynamics_radio') or ui_settings.get('evaluate_radio'):
            return SettingsFactory.create_analysis_settings(ui_settings)
        else:
            return SettingsFactory.create_reg_settings(ui_settings)
    
    def determine_p_mode(ui_settings: dict) -> str:
        mode_mapping = {
            'batch_radio': 'batch',
            'channel_align_radio': 'channel',
            'drift_radio': 'drift',
            'jitter_radio': 'jitter',
            'global_ref_radio': 'global',
            't_channel_align_radio': 't_channel'
        }
        
        for radio_name, mode in mode_mapping.items():
            if ui_settings.get(radio_name, False):
                return mode
                
        raise ValueError("No valid processing mode selected")

    def determine_reference_file(ui_settings: dict) -> Optional[Path]:
        """Determine reference file based on mode"""
        if ui_settings.get('batch_radio') or ui_settings.get('global_ref_radio'):
            ref_file = ui_settings.get('reference_file')
            return Path(ref_file) if ref_file else None
        return None

    def create_training_settings(ui_settings: dict) -> TrainingSettings:
        """Create training settings instance"""
        training_settings = {
            'input_path': ui_settings['input_path'],
            'output_path': ui_settings['output_path'],
            'reference_channel': ui_settings['reference_channel'],
            'p_mode': SettingsFactory.determine_p_mode(ui_settings),
            'learning_rate': ui_settings.get('learning_rate', 0.05),
            'target_loss': ui_settings.get('target_loss', 0.0001),
            'batch_size': ui_settings.get('batch_size', 4),
            'backward_batch_size': ui_settings.get('backward_batch_size', 4),
            'train_from_scratch': ui_settings.get('train_from_scratch', True),
            'model_path': ui_settings.get('model_path'),
            'sample_prefix': ui_settings.get('sample_prefix', ''),
            'channel_prefix': ui_settings.get('channel_prefix', ''),
            'time_prefix': ui_settings.get('time_prefix', ''),
            'reference_file': SettingsFactory.determine_reference_file(ui_settings),
            'window_size': ui_settings.get('window_size'),
            'ref_timepoint': ui_settings.get('ref_timepoint'),
            'reference_timepoint': ui_settings.get('reference_timepoint'),
            'is_timelapse': ui_settings.get('is_timelapse', False),
            'augment_data': ui_settings.get('augment_data', False),
            'augmentation_factor': ui_settings.get('augmentation_factor', 5),
            'max_rotation_angle': ui_settings.get('max_rotation_angle', 15.0),
            'max_translation_factor': ui_settings.get('max_translation_factor', 0.1),
            'is_unsupervised': ui_settings.get('autotraining_radio', False),
        }
        
        for key in ['input_path', 'output_path', 'model_path', 'reference_file']:
            if training_settings.get(key):
                training_settings[key] = Path(training_settings[key])
        return TrainingSettings(**training_settings)
            
    def create_reg_settings(ui_settings: dict) -> RegistrationSettings:
        """Create manual settings instance"""
        reg_settings = {
            'input_path': ui_settings['input_path'],
            'output_path': ui_settings['output_path'],
            'sample_prefix': ui_settings.get('sample_prefix', ''),
            'channel_prefix': ui_settings.get('channel_prefix', ''),
            'time_prefix': ui_settings.get('time_prefix', ''),
            'reference_channel': ui_settings['reference_channel'],
            'p_mode':SettingsFactory.determine_p_mode(ui_settings),
            'model_path': ui_settings.get('model_path'),
            'reference_file': SettingsFactory.determine_reference_file(ui_settings),
            'window_size': ui_settings.get('window_size'),
            'ref_timepoint': ui_settings.get('ref_timepoint'),
            'reference_timepoint': ui_settings.get('reference_timepoint'),
            'is_timelapse': ui_settings.get('is_timelapse', False)
        }
        
        for key in ['input_path', 'output_path', 'reference_file']:
            if reg_settings.get(key):
                reg_settings[key] = Path(reg_settings[key])
        return RegistrationSettings(**reg_settings)

    def create_analysis_settings(ui_settings: dict) -> AnalysisSettings:
        """Create analysis settings instance"""
        analysis_settings = {
            'model_path': ui_settings.get('model_path'),
            'output_path': ui_settings.get('output_path'),
            'input_path': ui_settings.get('input_path'),
            'reference_file': ui_settings.get('reference_file'),
            'analysis_mode': 'training_dynamics' if ui_settings.get('training_dynamics_radio') else 'evaluate',
            'p_mode': SettingsFactory.determine_p_mode(ui_settings),
            'reference_channel': ui_settings.get('reference_channel', 0),  
            'sample_prefix': ui_settings.get('sample_prefix', ''),
            'channel_prefix': ui_settings.get('channel_prefix', ''),
            'time_prefix': ui_settings.get('time_prefix', ''),
            'is_timelapse': ui_settings.get('is_timelapse', False)
        }
        
        # Add mode-specific settings
        if ui_settings.get('jitter_radio', False):
            analysis_settings['window_size'] = ui_settings.get('window_size')
            analysis_settings['ref_timepoint'] = ui_settings.get('ref_timepoint')
        
        if ui_settings.get('global_ref_radio', False):
            analysis_settings['reference_timepoint'] = ui_settings.get('reference_timepoint')
        
        # Convert paths to Path objects
        for key in ['model_path', 'output_path', 'input_path', 'reference_file']:
            if analysis_settings.get(key):
                analysis_settings[key] = Path(analysis_settings[key])
                
        return AnalysisSettings(**analysis_settings)

####################### Prep Datasets

@dataclass
class VisualizationData:
    """Data container for visualization parameters with correction_ui compatibility"""
    moving: Union[torch.Tensor, np.ndarray]  # Moving image
    reference: Union[torch.Tensor, np.ndarray]  # Reference image
    predicted_transform: torch.Tensor  # Current transform
    file_pair: Tuple[Union[str, Path], Union[str, Path]]  # (moving_file, reference_file)
    mode: str = 'correction'  # 'correction' or 'validate'
    
    def __post_init__(self):
        """Validate inputs and ensure compatibility"""
        # Validate mode
        if self.mode not in ['correction', 'validate']:
            raise ValueError("mode must be either 'correction' or 'validate'")

        # Ensure file_pair contains Path objects
        if isinstance(self.file_pair[0], str):
            self.file_pair = (Path(self.file_pair[0]), Path(self.file_pair[1]))
            
        # Ensure transform is a tensor
        if not isinstance(self.predicted_transform, torch.Tensor):
            self.predicted_transform = torch.tensor(self.predicted_transform, dtype=torch.float32)
    
    @staticmethod
    def _normalize_data(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize input data to proper format"""
        if isinstance(data, torch.Tensor):
            # Ensure proper dimensionality for visualization
            if data.dim() == 5:  # [B, C, D, H, W]
                data = data.squeeze(0).squeeze(0)  # Remove batch and channel dims
            elif data.dim() == 4:  # [B, D, H, W]
                data = data.squeeze(0)  # Remove batch dim
            return data
        elif isinstance(data, np.ndarray):
            # Ensure 3D array [D, H, W]
            if data.ndim == 5:
                data = data[0, 0]  # Remove batch and channel dims
            elif data.ndim == 4:
                data = data[0]  # Remove batch dim
            return data
        else:
            raise ValueError("Data must be either torch.Tensor or numpy.ndarray")

@dataclass
class ImagePair:
    """Container for a pair of images to be registered"""
    moving_path: Path
    reference_path: Path
    group_id: int
    channel: int
    timepoint: Optional[int] = None
    
    def load_images(self, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load both images as tensors"""
        moving = self.load_image(self.moving_path, normalize)
        reference = self.load_image(self.reference_path, normalize)
        return moving, reference
    
    @staticmethod
    def load_image(path: Path, normalize: bool = True) -> torch.Tensor:
        """Load single image as tensor with proper normalization"""
        img = tifffile.imread(str(path))
        if normalize:
            img = img.astype(np.float32)
            if img.dtype == np.uint8:
                img = img / 255.0
            elif img.dtype == np.uint16:
                img = img / 65535.0
        tensor = torch.from_numpy(img).float()
        # Ensure 5D tensor [B, C, D, H, W]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        return tensor
