import numpy as np
import os
from tifffile import TiffFile, imwrite
import matplotlib.pyplot as plt

# Hardcoded paths and options
input_cells = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/E2/E2_cell_data_features.npy'
reference_stack = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1.tif'
output_dir = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/E2/'
    

def get_metadata(file_path):
    """Extract metadata from a TIFF file for proper stack creation."""
    tif = TiffFile(file_path)
    imagej = tif.imagej_metadata
    
    # Image Array
    data = tif.asarray()
    
    # Get dimensions
    dimension = data.shape
    dimension_order = tif.series[0].axes
    
    # Map dimension names to their sizes
    dimension_sizes = {}
    for i in range(len(dimension)):
        dimension_sizes[dimension_order[i]] = dimension[i]
    
    size_x = dimension_sizes.get('X', 'N/A')
    size_y = dimension_sizes.get('Y', 'N/A')
    size_z = dimension_sizes.get('Z', 'N/A')
    
    # Get pixel size
    x = tif.pages[0].tags['XResolution'].value
    pixel_size_x = x[1] / x[0]
    y = tif.pages[0].tags['YResolution'].value
    pixel_size_y = y[1] / y[0]
    pixel_size_z = imagej.get('spacing', 1.0)
    
    # Get unit
    physical_size_unit = imagej.get('unit', 'um')
    
    # Get pixel type
    pixel_type = data.dtype
    
    return pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order

def create_intensity_stack(cell_data, reference_stack_path, output_path, feature_name):

    print(f"Creating intensity stack for: {feature_name}")
    
    # Retrieve metadata about the TIFF file to ensure accurate dimensioning and scaling
    pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order = get_metadata(reference_stack_path)
    shape = (size_z, size_y, size_x)
    
    # Initialize the output array
    intensity_array = np.zeros(shape, dtype=np.uint8)
    
    # Determine the maximum value for the specified feature for normalization
    max_value = max((cell.get(feature_name, 0) for cell in cell_data if feature_name in cell), default=0)
    
    print(f"Feature range: 0 to {max_value:.3f}")
    
    # Set voxel values based on cell feature values
    print(f"Mapping features to voxels...")
    for cell in cell_data:
        if feature_name in cell and max_value > 0:
            # Normalize feature to 0-255 range
            intensity = int(255 * (cell[feature_name] / max_value))
            
            # Get cell coordinates
            if 'coordinates' not in cell:
                continue
                
            coords = cell['coordinates']
            
            # Handle different coordinate formats
            if isinstance(coords, tuple) and len(coords) == 3:
                # Format: (z_coords, y_coords, x_coords)
                z_coords, y_coords, x_coords = coords
            elif isinstance(coords, list):
                # Format: list of (z, y, x) tuples
                z_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                x_coords = [c[2] for c in coords]
            else:
                print(f"Warning: Unsupported coordinates format, skipping")
                continue
            
            # Set voxel values
            for i in range(len(x_coords)):
                x, y, z = x_coords[i], y_coords[i], z_coords[i]
                
                # Check bounds
                if 0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z:
                    intensity_array[z, y, x] = intensity
    
    # Save the intensity map as a TIFF file with appropriate metadata
    print(f"Saving intensity stack to: {output_path}")
    imwrite(output_path, intensity_array, imagej=True, 
            resolution=(1./pixel_size_x, 1./pixel_size_y), 
            metadata={'spacing': pixel_size_z, 'unit': physical_size_unit, 'axes': 'ZYX'})
    
    return output_path

def generate_gradient_feature_stacks(cell_data, reference_stack_path, output_dir):

    # Skip these keys that aren't scalar features
    skip_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 
                    'spike_heights', 'spike_widths', 'spike_amplitudes', 'sync_community']
    
    # Get all keys from the first cell
    first_cell = cell_data[0]
    
    features_to_process = []
    for key, value in first_cell.items():
        # Skip non-scalar values and keys in skip_keys
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        
        # Check if the value is a scalar number
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            features_to_process.append(key)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Available features: {features_to_process}")
    
    if not features_to_process:
        print("No features to process!")
    
    # Create stacks for each feature
    output_paths = {}
    
    for feature in features_to_process:
        output_path = os.path.join(output_dir, f"{feature}_map.tif")
        
        create_intensity_stack(
            cell_data=cell_data,
            reference_stack_path=reference_stack_path,
            output_path=output_path,
            feature_name=feature
        )
        
        output_paths[feature] = output_path
    
    return output_paths

def create_binary_match_stacks(cell_data, reference_stack_path, output_dir, name_patterns=None):

    if name_patterns is None:
        name_patterns = ["best_match_Lineage", "best_match_Dependancy", "best_match_Time", "sync_community"]
    
    print(f"Creating binary stacks for unique values in features matching: {name_patterns}")
    
    # Get metadata from the reference stack
    pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order = get_metadata(reference_stack_path)
    stack_shape = (size_z, size_y, size_x)
    
    # Find all matching features across all cells
    matching_features = []
    for cell in cell_data:
        for key in cell.keys():
            if any(pattern in key for pattern in name_patterns):
                matching_features.append(key)
    
    matching_features = sorted(list(set(matching_features)))
    print(f"Found {len(matching_features)} matching features: {matching_features}")
    
    # Collect all unique values for each matching feature
    value_to_feature = {}  # Maps each unique value to its parent feature
    for feature in matching_features:
        for cell in cell_data:
            if feature in cell and cell[feature]:
                # Get the value as a string for consistent handling
                value = str(cell[feature])
                if value:  # Skip empty values
                    value_to_feature[value] = feature
    
    print(f"Found {len(value_to_feature)} unique values across matching features")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # For each unique value, create a binary stack
    for value, parent_feature in value_to_feature.items():

        output_path = os.path.join(output_dir, f"{value}.tif")
        print(f"Creating binary stack for value: {value} from feature: {parent_feature}")
        
        # Initialize the output array
        binary_array = np.zeros(stack_shape, dtype=np.uint8)
        
        # Count cells with this value for reporting
        cell_count = 0
        
        # Set voxel values for cells that have this value
        for cell in cell_data:
            if parent_feature in cell and str(cell[parent_feature]) == value:
                cell_count += 1
                
                # Get cell coordinates
                if 'coordinates' not in cell:
                    continue
                
                coords = cell['coordinates']
                
                # Handle different coordinate formats
                if isinstance(coords, tuple) and len(coords) == 3:
                    # Format: (z_coords, y_coords, x_coords)
                    z_coords, y_coords, x_coords = coords
                elif isinstance(coords, list):
                    # Format: list of (z, y, x) tuples
                    z_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    x_coords = [c[2] for c in coords]
                else:
                    print(f"Warning: Unsupported coordinates format, skipping")
                    continue
                
                # Set voxel values to 255 (full intensity)
                for i in range(len(x_coords)):
                    x, y, z = x_coords[i], y_coords[i], z_coords[i]
                    
                    # Check bounds
                    if 0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z:
                        binary_array[z, y, x] = 255
        
        print(f"Marked {cell_count} cells with value '{value}'")
        
        # Save the binary map as a TIFF file with appropriate metadata
        print(f"Saving binary stack to: {output_path}")
        imwrite(output_path, binary_array, imagej=True,
                resolution=(1./pixel_size_x, 1./pixel_size_y),
                metadata={'spacing': pixel_size_z, 'unit': physical_size_unit, 'axes': 'ZYX'})
        
        output_paths[value] = output_path
    
    return output_paths

if __name__ == '__main__':
    
    # Load cell data
    print(f"Loading cell data from: {input_cells}")
    cell_data = np.load(input_cells, allow_pickle=True)
    print(f"Loaded {len(cell_data)} cells")
    
    # Generate feature maps
    generate_gradient_feature_stacks(
            cell_data=cell_data,
            reference_stack_path=reference_stack,
            output_dir=output_dir)
    
    # Generate binary match stacks
    create_binary_match_stacks(
        cell_data=cell_data,
        reference_stack_path=reference_stack,
        output_dir=output_dir
    )

    print("Stack generation complete!")