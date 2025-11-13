import numpy as np
import os
from tifffile import TiffFile, imwrite
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def create_spatial_index(cells, cell_size=20):
    """
    Create a spatial index (grid-based) for fast cell overlap detection
    
    Args:
        cells: List of cell dictionaries
        cell_size: Size of grid cells for spatial indexing
        
    Returns:
        Dictionary mapping grid coordinates to lists of cell indices
    """
    spatial_index = {}
    
    for cell_idx, cell in enumerate(cells):
        coords = cell['coordinates']
        # Get bounding box
        min_z, max_z = np.min(coords[0]), np.max(coords[0])
        min_y, max_y = np.min(coords[1]), np.max(coords[1])
        min_x, max_x = np.min(coords[2]), np.max(coords[2])
        
        # Convert to grid coordinates
        grid_min_z, grid_max_z = min_z // cell_size, max_z // cell_size
        grid_min_y, grid_max_y = min_y // cell_size, max_y // cell_size
        grid_min_x, grid_max_x = min_x // cell_size, max_x // cell_size
        
        # Add cell to all grid cells it intersects
        for gz in range(grid_min_z, grid_max_z + 1):
            for gy in range(grid_min_y, grid_max_y + 1):
                for gx in range(grid_min_x, grid_max_x + 1):
                    grid_key = (gz, gy, gx)
                    if grid_key not in spatial_index:
                        spatial_index[grid_key] = []
                    spatial_index[grid_key].append(cell_idx)
    
    return spatial_index

def load_cell_data(filepaths):
    """
    Load cell data from multiple .npy files and pre-compute coordinate sets
    """
    all_samples = []
    for filepath in filepaths:
        cells = np.load(filepath, allow_pickle=True)
        # Pre-compute coordinate sets for each cell
        for cell in cells:
            cell['coords_set'] = set(zip(*cell['coordinates']))
        all_samples.append(cells)
    return all_samples

def calculate_overlap_for_pair(data):
    """
    Calculate overlap between a cell and a merged cell (for parallel processing)
    
    Args:
        data: Tuple containing (cell_coords, merged_cell_coords, cell_key, merged_cell_idx)
              where cell_key is (sample_idx, cell_idx)
        
    Returns:
        Tuple of (cell_key, merged_cell_idx, overlap_percentage)
    """
    cell_coords, merged_cell_coords, cell_key, merged_cell_idx = data
    
    # Convert coordinates to sets of tuples for comparison
    if isinstance(cell_coords, dict) and 'coords_set' in cell_coords:
        coords_set1 = cell_coords['coords_set']
    else:
        coords_set1 = set(zip(*cell_coords))
    
    if isinstance(merged_cell_coords, dict) and 'coords_set' in merged_cell_coords:
        coords_set2 = merged_cell_coords['coords_set']
    else:
        coords_set2 = set(zip(*merged_cell_coords))
    
    if not coords_set1 or not coords_set2:
        return cell_key, merged_cell_idx, 0
    
    intersection = coords_set1.intersection(coords_set2)
    smaller_set_size = min(len(coords_set1), len(coords_set2))
    
    if smaller_set_size == 0:
        return cell_key, merged_cell_idx, 0
    
    overlap = len(intersection) / smaller_set_size
    return cell_key, merged_cell_idx, overlap

def calculate_overlap(coords1, coords2):
    """
    Calculate overlap between two sets of coordinates more efficiently
    """
    # Convert coordinates to sets of tuples if they're not already
    if isinstance(coords1, tuple) and len(coords1) == 3:
        coords_set1 = set(zip(*coords1))
    else:
        coords_set1 = coords1
        
    if isinstance(coords2, tuple) and len(coords2) == 3:
        coords_set2 = set(zip(*coords2))
    else:
        coords_set2 = coords2
    
    if not coords_set1 or not coords_set2:
        return 0
    
    # Use NumPy operations for set intersection
    intersection = coords_set1.intersection(coords_set2)
    smaller_set_size = min(len(coords_set1), len(coords_set2))
    
    if smaller_set_size == 0:
        return 0
    
    return len(intersection) / smaller_set_size

def merge_cell_dictionaries(cells):
    """
    Merge multiple cell dictionaries by averaging their data
    """
    # Create a new dictionary for the merged cell
    merged_cell = {}
    
    # The cell_id will be assigned by the caller
    
    # Merge coordinates (take the union of all voxels)
    all_coords_sets = [set(zip(*cell['coordinates'])) for cell in cells]
    union_coords = set().union(*all_coords_sets)
    
    # Convert back to the original format (tuple of arrays)
    z_coords, y_coords, x_coords = zip(*union_coords)
    merged_cell['coordinates'] = (np.array(z_coords), np.array(y_coords), np.array(x_coords))
    
    # Average the tracks if they have the same length
    track_lengths = [len(cell['track']) for cell in cells]
    if len(set(track_lengths)) == 1:  # All tracks have the same length
        merged_tracks = []
        track_length = track_lengths[0]
        
        for i in range(track_length):
            avg_value = np.mean([cell['track'][i] for cell in cells])
            merged_tracks.append(avg_value)
        
        merged_cell['track'] = merged_tracks
    else:
        # If tracks have different lengths, keep the track from the largest cell
        largest_cell_idx = np.argmax([len(set(zip(*cell['coordinates']))) for cell in cells])
        merged_cell['track'] = cells[largest_cell_idx]['track']
    
    # Add occurrence count
    merged_cell['occurrence_count'] = len(cells)
    
    return merged_cell

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

def generate_mask_from_cells(cells, reference_mask_path, output_file):
    """
    Generate a new mask from the merged cells with proper metadata
    """
    # Get metadata from the reference mask
    pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order = get_metadata(reference_mask_path)
    
    # Create an empty mask with the same shape as the reference mask
    shape = (size_z, size_y, size_x)
    mask = np.zeros(shape, dtype=np.uint16)  # Use uint16 for cell_id values
    
    # Fill the mask with cell_ids
    for cell in cells:
        coords = cell['coordinates']
        mask[coords] = cell['cell_id']
    
    # Save the mask with proper metadata
    imwrite(output_file, mask, imagej=True, 
            resolution=(1./pixel_size_x, 1./pixel_size_y), 
            metadata={'spacing': pixel_size_z, 'unit': physical_size_unit, 'axes': 'ZYX'})

def merge_cell_masks(sample_filepaths, output_npy, output_mask, reference_mask_path, n_workers=None):
    """
    Merge cell masks from multiple samples using parallel processing and spatial indexing
    """
    start_time = time.time()
    
    # Load cell data from all samples
    all_samples = load_cell_data(sample_filepaths)
    
    # Create a master list for merged cells
    merged_cells = []
    
    # Track processed cells with a more efficient structure
    processed_cell_indices = set()
    
    # Track the next available unique cell_id
    next_cell_id = 1  # Start with ID 1
    
    # Set the batch size for parallel processing
    batch_size = 100  # Adjust based on your system's capabilities
    
    # Process each sample
    for sample_idx, sample_cells in enumerate(all_samples):
        print(f"Processing sample {sample_idx+1}/{len(all_samples)}")
        
        # Create spatial index for current merged cells
        spatial_index = create_spatial_index(merged_cells)
        
        # Process cells in batches for better parallel efficiency
        for batch_start in range(0, len(sample_cells), batch_size):
            batch_end = min(batch_start + batch_size, len(sample_cells))
            batch = sample_cells[batch_start:batch_end]
            
            # Create tasks for parallel processing
            cell_tasks = []
            for batch_idx, cell in enumerate(batch):
                cell_idx = batch_start + batch_idx
                if (sample_idx, cell_idx) in processed_cell_indices:
                    continue
                
                # Find potential overlapping cells using spatial index
                potential_matches = set()
                coords = cell['coordinates']
                min_z, max_z = np.min(coords[0]), np.max(coords[0])
                min_y, max_y = np.min(coords[1]), np.max(coords[1])
                min_x, max_x = np.min(coords[2]), np.max(coords[2])
                
                # Convert to grid coordinates
                cell_size = 20
                grid_min_z, grid_max_z = min_z // cell_size, max_z // cell_size
                grid_min_y, grid_max_y = min_y // cell_size, max_y // cell_size
                grid_min_x, grid_max_x = min_x // cell_size, max_x // cell_size
                
                # Check all grid cells the cell might overlap with
                for gz in range(grid_min_z, grid_max_z + 1):
                    for gy in range(grid_min_y, grid_max_y + 1):
                        for gx in range(grid_min_x, grid_max_x + 1):
                            grid_key = (gz, gy, gx)
                            if grid_key in spatial_index:
                                potential_matches.update(spatial_index[grid_key])
                
                # Only check potential matches instead of all merged cells
                for merged_idx in potential_matches:
                    cell_tasks.append((cell['coordinates'], merged_cells[merged_idx]['coordinates'], 
                                      (sample_idx, cell_idx), merged_idx))
            
            # Process overlap checks in parallel
            overlaps = {}  # Map (sample_idx, cell_idx) to list of (merged_idx, overlap) pairs
            
            if cell_tasks:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(calculate_overlap_for_pair, task) for task in cell_tasks]
                    
                    for future in as_completed(futures):
                        cell_key, merged_idx, overlap = future.result()
                        if overlap > 0.5:  # Significant overlap
                            if cell_key not in overlaps:
                                overlaps[cell_key] = []
                            overlaps[cell_key].append((merged_idx, overlap))
            
            # Process overlap results and merge cells
            for batch_idx, cell in enumerate(batch):
                cell_idx = batch_start + batch_idx
                cell_key = (sample_idx, cell_idx)
                
                if cell_key in processed_cell_indices:
                    continue
                
                if cell_key in overlaps and overlaps[cell_key]:
                    # Sort by overlap to prioritize best matches
                    overlapping_merged_indices = [pair[0] for pair in 
                                                sorted(overlaps[cell_key], key=lambda x: x[1], reverse=True)]
                    
                    # Collect cells to merge
                    overlapping_cells = [merged_cells[idx] for idx in overlapping_merged_indices]
                    all_cells_to_merge = overlapping_cells + [cell]
                    
                    # Merge cells
                    merged_cell = merge_cell_dictionaries(all_cells_to_merge)
                    # Assign the next unique cell_id
                    merged_cell['cell_id'] = next_cell_id
                    next_cell_id += 1
                    
                    # Remove overlapping cells from the master list in reverse order
                    for idx in sorted(overlapping_merged_indices, reverse=True):
                        merged_cells.pop(idx)
                    
                    # Add the merged cell
                    merged_cells.append(merged_cell)
                else:
                    # No significant overlap, add the cell as is but with a unique ID
                    new_cell = cell.copy()
                    # Preserve all properties except cell_id and add occurrence count
                    new_cell['cell_id'] = next_cell_id
                    next_cell_id += 1
                    new_cell['occurrence_count'] = 1
                    merged_cells.append(new_cell)
                
                # Mark as processed
                processed_cell_indices.add(cell_key)
            
            # Update spatial index after processing a batch
            spatial_index = create_spatial_index(merged_cells)
            
            elapsed = time.time() - start_time
            print(f"  Processed batch {batch_start}-{batch_end}/{len(sample_cells)} in {elapsed:.2f} seconds")
    
    # Remove pre-computed coordinate sets before saving
    for cell in merged_cells:
        if 'coords_set' in cell:
            del cell['coords_set']
    
    # Save the merged cell data
    np.save(output_npy, merged_cells)
    
    # Generate the merged mask with proper metadata
    generate_mask_from_cells(merged_cells, reference_mask_path, output_mask)
    
    total_time = time.time() - start_time
    print(f"Merged {sum(cell['occurrence_count'] for cell in merged_cells)} cells into {len(merged_cells)} unique cells")
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    # Paths to input cell data
    sample_filepaths = [
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1_cell_tracks_and_coordinates.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E2/E2_cell_tracks_and_coordinates.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E3/E3_cell_tracks_and_coordinates.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E4/E4_cell_tracks_and_coordinates.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E5/E5_cell_tracks_and_coordinates.npy',
    ]
    
    # Output paths
    output_npy = "F:/02-Recording/02-Calcium/30min_ScrambledKO/Avg/merged_cell_data.npy"
    output_mask = "F:/02-Recording/02-Calcium/30min_ScrambledKO/Avg/merged_mask.tif"
    
    # Path to a reference mask (to get the shape and metadata)
    reference_mask_path = "F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/MAXT_E1.tif"

    # Number of parallel workers (None = auto-detect based on CPU cores)
    n_workers = None

    # Merge the cell masks
    merge_cell_masks(sample_filepaths, output_npy, output_mask, reference_mask_path, n_workers)