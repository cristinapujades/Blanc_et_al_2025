import numpy as np
from pathlib import Path
import tifffile
import re
from multiprocessing import Pool, cpu_count

def process_cell(args):
    """
    Process a single cell by directly checking mask2 values at cell coordinates.
    Parameters:
        args: tuple (label1, mask1, mask2)
            label1 (int): Cell label from mask1
            mask1 (np.ndarray): Primary mask array
            mask2 (np.ndarray): Reference mask array
    Returns:
        tuple: (label1, should_remove)
            should_remove (bool): True if cell overlaps above threshold
    """
    label1, mask1, mask2 = args
    
    # Get cell coordinates and corresponding mask2 values
    cell_coords = np.where(mask1 == label1)
    mask2_values = mask2[cell_coords]
    
    # Count total voxels and overlapping voxels
    total_voxels = len(mask2_values)
    overlap_voxels = np.sum(mask2_values > 0)
    
    # Compute overlap fraction and determine if cell should be removed
    overlap_fraction = overlap_voxels / total_voxels
    return label1, overlap_fraction >= 0.1

def compute_contextualized_mask(mask1: np.ndarray, mask2: np.ndarray, 
                              n_processes: int = None) -> np.ndarray:
    """
    Generate contextualized mask using parallel direct voxel access.
    Parameters:
        mask1 (np.ndarray): Primary mask (16-bit integer array)
        mask2 (np.ndarray): Reference mask (16-bit integer array)
        n_processes (int): Number of processes to use
    Returns:
        np.ndarray: Contextualized mask (16-bit integer array)
    """
    # Input validation
    assert mask1.dtype in (np.uint16, np.int16), f"Mask1 dtype is {mask1.dtype}, expected uint16/int16"
    assert mask2.dtype in (np.uint16, np.int16), f"Mask2 dtype is {mask2.dtype}, expected uint16/int16"
    
    # Get unique cell labels
    labels1 = np.unique(mask1[mask1 > 0])
    
    # Prepare arguments for parallel processing
    args = [(label, mask1, mask2) for label in labels1]
    
    with Pool(n_processes) as pool:
        results = pool.map(process_cell, args)
    
    # Generate output mask
    contextualized = mask1.copy()
    for label, should_remove in results:
        if should_remove:
            contextualized[mask1 == label] = 0
    
    return contextualized

def process_directory(input_dir: str, output_dir: str, n_processes: int = None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find and group files
    files = list(input_path.glob('*_cp_masks.tif'))
    experiments = {}
    
    for file in files:
        match = re.search(r'C([23])E(\d+)', file.name)
        if match:
            channel, exp_num = match.groups()
            if exp_num not in experiments:
                experiments[exp_num] = {}
            experiments[exp_num][f'C{channel}'] = file
    
    # Process each experiment
    for exp_num, pair in experiments.items():
        if 'C2' in pair and 'C3' in pair:
            # Load masks
            mask1 = tifffile.imread(pair['C2'])
            mask2 = tifffile.imread(pair['C3'])
            
            # Generate contextualized mask
            contextualized = compute_contextualized_mask(mask1, mask2, n_processes)
            
            # Save output
            output_file = output_path / f'C2E{exp_num}.tif'
            tifffile.imwrite(str(output_file), contextualized)

if __name__ == "__main__":
    input_dir = 'F:/01-Analyzed/03-Segmentation/Ngn1Cre x vglut2aSwitch/Neurog1KO/'
    output_dir = 'F:/01-Analyzed/04-Contextualized/Ngn1Cre x vglut2aSwitch/Neurog1KO/'
    n_processes = cpu_count()-2  
    
    process_directory(input_dir, output_dir, n_processes)