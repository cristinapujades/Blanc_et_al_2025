from dataclasses import dataclass
from typing import Optional, Dict, List
import re
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QObject

def retrieve_counts(organized_files: Dict[int, Dict[int, Dict[int, List[Path]]]]):
    # Retrieve number of samples
    num_samples = len(organized_files)

    # Retrieve number of timepoints per sample
    num_timepoints_per_sample = {sample: len(timepoints) for sample, timepoints in organized_files.items()}

    # Retrieve number of channels per timepoint for each sample
    num_channels_per_timepoint = {
        sample: {timepoint: len(channels) for timepoint, channels in timepoints.items()}
        for sample, timepoints in organized_files.items()
    }

    return num_samples, num_timepoints_per_sample, num_channels_per_timepoint

@dataclass
class FileSorting(QObject):
    """Advanced settings for flexible file naming pattern recognition."""
    sample_prefix: Optional[str] = None
    time_prefix: Optional[str] = None
    channel_prefix: Optional[str] = None
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, settings):
        super().__init__()
        self.input_path = Path(settings.input_path)
        self.sample_prefix = getattr(settings, 'sample_prefix', None)
        self.time_prefix = getattr(settings, 'time_prefix', None) 
        self.channel_prefix = getattr(settings, 'channel_prefix', None)
        self._process_and_validate_prefixes()
        self._generate_patterns()

    def _process_and_validate_prefixes(self):
        """Clean, validate, and normalize user-defined prefixes in one function."""
        def normalize_prefix(prefix: Optional[str]) -> Optional[str]:
            """Normalize prefix input by stripping spaces and converting to uppercase."""
            return prefix.strip().upper() if prefix else None

        # Normalize prefixes
        self.sample_prefix = normalize_prefix(self.sample_prefix)
        self.time_prefix = normalize_prefix(self.time_prefix)
        self.channel_prefix = normalize_prefix(self.channel_prefix)

        # Validate uniqueness of prefixes
        provided_prefixes = {p for p in [self.sample_prefix, self.time_prefix, self.channel_prefix] if p}
        if len(provided_prefixes) != len(set(provided_prefixes)):
            raise ValueError("Each provided prefix must be unique.")

    def _generate_patterns(self):
        """Generate regex patterns only for specified prefixes"""
        def find_pattern(prefix):
            if prefix is None:
                return None
            pattern = rf'{re.escape(str(prefix))}(\d+)'
            return pattern

        self.sample_pattern = find_pattern(self.sample_prefix)
        self.time_pattern = find_pattern(self.time_prefix)
        self.channel_pattern = find_pattern(self.channel_prefix)

    def organize_files(self) -> Dict[int, Dict[int, Dict[int, List[Path]]]]:
        """
        Organize files into a nested structure based on sample, timepoint, and channel with file paths
        """
        all_files = list(self.input_path.glob('*.tif*'))
        if not all_files:
            self.progress_signal.emit("No .tif files found in the input directory")
            return {}

        # Initialize with default value 1 if no prefix specified
        sample_values = {1} if not self.sample_prefix else set()
        timepoint_values = {1} if not self.time_prefix else set()
        channel_values = {1} if not self.channel_prefix else set()

        # Only collect values for specified prefixes
        for file_path in all_files:
            filename = file_path.name
            
            if self.sample_prefix:
                sample_match = re.search(self.sample_pattern, filename, re.IGNORECASE)
                if sample_match:
                    sample_values.add(int(sample_match.group(1)))
            
            if self.time_prefix:
                time_match = re.search(self.time_pattern, filename, re.IGNORECASE)
                if time_match:
                    timepoint_values.add(int(time_match.group(1)))
            
            if self.channel_prefix:
                channel_match = re.search(self.channel_pattern, filename, re.IGNORECASE)
                if channel_match:
                    channel_values.add(int(channel_match.group(1)))

        # Sort all collected values
        samples = sorted(sample_values)
        timepoints = sorted(timepoint_values)
        channels = sorted(channel_values)

        # Initialize the structure
        organized_files: Dict[int, Dict[int, Dict[int, List[Path]]]] = {}
        
        # Create the nested structure
        for sample in samples:
            organized_files[sample] = {}
            for timepoint in timepoints:
                organized_files[sample][timepoint] = {}
                for channel in channels:
                    organized_files[sample][timepoint][channel] = []

        # Populate with files based on prefix patterns
        for file_path in all_files:
            filename = file_path.name
            
            # Get values, defaulting to first value if no prefix or no match
            sample = (int(match.group(1)) if (self.sample_prefix and 
                    (match := re.search(self.sample_pattern, filename, re.IGNORECASE)))
                    else samples[0])
            
            timepoint = (int(match.group(1)) if (self.time_prefix and 
                       (match := re.search(self.time_pattern, filename, re.IGNORECASE)))
                       else timepoints[0])
            
            channel = (int(match.group(1)) if (self.channel_prefix and 
                     (match := re.search(self.channel_pattern, filename, re.IGNORECASE)))
                     else channels[0])
            
            # Add file to appropriate location
            organized_files[sample][timepoint][channel].append(file_path)

        self.progress_signal.emit(f"Found {len(all_files)} files across {len(samples)} samples, "
                            f"{len(timepoints)} timepoints, and {len(channels)} channels")
        self.finished_signal.emit()
        return organized_files
