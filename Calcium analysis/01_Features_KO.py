import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq
import networkx as nx
import community.community_louvain as community_louvain
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import os
import time
from tifffile import imread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Input and output paths
input_file = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/E2/E2_cell_tracks_and_coordinates.npy'
output_file = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/E2/E2_cell_data_features.npy'
output_pdf = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/E2/E2_cell_tracks_spikes.pdf'

Lineage_stacks = {
         'gad': 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/00-Lineage/Gad.tif',
         'glut': 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_Neurog1KO/00-Lineage/Glut.tif',
    }

# Configuration parameters
SMOOTHING_SIGMA = 1.5  # Gaussian smoothing parameter
WINDOW_SIZE = 50 # Window size for dynamic threshold calculation
STD_FACTOR = 1.8  # Standard deviation multiplier for spike detection
PROMINENCE = 1  # Prominence parameter for spike detection
WIDTH = 1  # Width parameter for spike detection
SAMPLING_RATE = 2.0  # Data acquisition rate in Hz for Fourier analysis

##############################################################################

def detect_spikes(track, window_size=WINDOW_SIZE, std_factor=STD_FACTOR, prominence=PROMINENCE, width=WIDTH):
    if len(track) < window_size:
        return {
            'num_spikes': 0, 
            'peak_indices': np.array([]), 
            'spike_heights': np.array([]),
            'spike_widths': np.array([]),
            'average_spike_width': 0,
            'average_spike_amplitude': 0
        }
    
    # Compute the moving average
    moving_avg = np.convolve(track, np.ones(window_size) / window_size, mode='same')
    
    # Compute moving standard deviation
    squared_diff = (track - moving_avg) ** 2
    moving_var = np.convolve(squared_diff, np.ones(window_size) / window_size, mode='same')
    moving_std = np.sqrt(moving_var)
    
    # Calculate dynamic threshold
    dynamic_threshold = moving_avg + std_factor * moving_std
    
    # Find peaks using the dynamic threshold
    peaks, properties = find_peaks(
        track, 
        height=dynamic_threshold, 
        prominence=prominence, 
        width=width)
    
    # Rest of the function remains the same...
    spike_heights = track[peaks] if peaks.size > 0 else np.array([])
    spike_widths = properties.get('widths', np.array([]))
    
    if len(peaks) > 0:
        baseline_heights = moving_avg[peaks]
        spike_amplitudes = spike_heights - baseline_heights
        average_spike_amplitude = np.mean(spike_amplitudes)
    else:
        spike_amplitudes = np.array([])
        average_spike_amplitude = 0
    
    average_spike_width = np.mean(spike_widths) if spike_widths.size > 0 else 0
    
    return {
        'num_spikes': len(peaks), 
        'average_spike_width': average_spike_width,
        'average_spike_amplitude': average_spike_amplitude,
        'peak_indices': peaks
    }

def extract_features_fourier(signal, sampling_rate=SAMPLING_RATE):
    """Extract simplified frequency domain features from calcium imaging signal.
    
    Args:
        signal (array): Time series signal
        sampling_rate (float): Data acquisition rate in Hz
        
    Returns:
        dict: Dictionary containing frequency centroid and spectral entropy
    """
    # Apply Hanning window to reduce spectral leakage
    windowed_signal = signal * np.hanning(len(signal))
    
    # Compute the FFT and frequency axis
    n = len(windowed_signal)
    yf = fft(windowed_signal)
    xf = fftfreq(n, 1 / sampling_rate)
    
    # Consider only positive frequencies up to Nyquist frequency
    pos_mask = xf > 0
    pos_freqs = xf[pos_mask]
    amplitude_spectrum = np.abs(yf[pos_mask])
    
    # Compute power spectrum (squared amplitude)
    power_spectrum = amplitude_spectrum**2
    
    # Calculate total power
    total_power = np.sum(power_spectrum)
    
    # Initialize default values for features
    frequency_centroid = 0
    spectral_entropy = 0
    
    if total_power > 0 and len(pos_freqs) > 0:
        # Frequency centroid (weighted average frequency)
        frequency_centroid = np.sum(pos_freqs * power_spectrum) / total_power
        
        # Spectral entropy (measure of regularity/complexity)
        norm_power = power_spectrum / total_power
        spectral_entropy = -np.sum(norm_power * np.log2(norm_power + 1e-9))
    
    # Return only the selected features
    return {
        'frequency_avg': frequency_centroid,
        'spectral_entropy': spectral_entropy
    }

#############################################################################

def spatial_correlation(cell_data, reference_stack_path, feature_name='reference_score'):
    """
    Evaluates cells against a reference TIFF stack to determine matching scores.
    
    For each cell, this function:
    1. Calculates the percentage of the cell's voxels that have non-zero values in the reference stack
    2. Calculates the mean intensity of matching voxels
    3. Generates a matching score based on percentage of matches and intensity

    """
    # Load the reference stack
    print(f"Loading reference stack: {reference_stack_path}")
    reference_stack = imread(reference_stack_path)

    print(f"Reference stack shape: {reference_stack.shape}")
    
    # Reference stack is in [z, y, x] order
    z_max, y_max, x_max = reference_stack.shape
    print(f"Reference stack dimensions: z={z_max}, y={y_max}, x={x_max}")

    # Evaluate intensity range
    non_zero_values = reference_stack[reference_stack > 0]
    if len(non_zero_values) > 0:
        min_intensity = np.min(non_zero_values)
        max_intensity = np.max(non_zero_values)
        print(f"Reference stack intensity range: {min_intensity} to {max_intensity}")
    else:
        print("Warning: Reference stack contains no non-zero values")
        return cell_data
    
    # Process each cell
    for cell_idx, cell in enumerate(cell_data):
        if cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{len(cell_data)}")

        if 'coordinates' not in cell:
            print(f"Warning: Cell {cell_idx} has no coordinates, skipping")
            continue

        # Extract coordinates
        coords = cell['coordinates']

        if isinstance(coords, tuple) and len(coords) == 3:
            # Format: (z_coords, y_coords, x_coords)
            z_coords, y_coords, x_coords = coords
        elif isinstance(coords, list):
            # Format: list of (z, y, x) tuples
            z_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            x_coords = [c[2] for c in coords]
        else:
            print(f"Warning: Unsupported coordinates format for cell {cell_idx}, skipping")
            continue

        # Calculate total number of voxels for this cell
        total_voxels = len(x_coords)
        if total_voxels == 0:
            print(f"Warning: Cell {cell_idx} has no coordinates, skipping")
            continue

        # Check each coordinate against the reference stack
        matching_voxels = 0
        matching_intensities = []

        for i in range(total_voxels):
            # Get coordinates in the correct order
            z, y, x = z_coords[i], y_coords[i], x_coords[i]
            
            # Strict bounds checking
            if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
                # Access with [z, y, x] order
                intensity = reference_stack[z, y, x]
                
                # Check if this is a positive voxel
                if intensity > 0:
                    matching_voxels += 1
                    matching_intensities.append(intensity)
        
        # Calculate matching statistics
        match_percentage = (matching_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        
        if matching_voxels > 0:
            mean_match_intensity = np.mean(matching_intensities)
            # Normalize mean intensity to 0-100 range
            norm_intensity = ((mean_match_intensity - min_intensity) / 
                            (max_intensity - min_intensity) * 100) if max_intensity > min_intensity else 0
            
            # Calculate combined score (50% weight to match percentage, 50% to intensity)
            combined_score = (match_percentage * 0.5) + (norm_intensity * 0.5)
        else:
            combined_score = 0
        
        # Add feature to cell dictionary
        cell[f'{feature_name}_match'] = combined_score
    
    return cell_data

def determine_best_match(cell_data, feature_stacks, feature_name, min_score_threshold=0):
    # Get reference names from the stacks dictionary
    reference_names = list(feature_stacks.keys())
    
    print(f"Determining best match from references: {reference_names} (minimum threshold: {min_score_threshold})")
    
    for cell_idx, cell in enumerate(cell_data):
        if cell_idx % 1000 == 0 and cell_idx > 0:
            print(f"Processing best match for cell {cell_idx}/{len(cell_data)}")
        
        # Collect scores for each reference
        match_scores = {}
        for ref_name in reference_names:
            score_key = f'{ref_name}_match'
            if score_key in cell and cell[score_key] > min_score_threshold:
                match_scores[ref_name] = cell[score_key]
        
        # Determine best match
        if match_scores:
            best_match = max(match_scores.items(), key=lambda x: x[1])
            cell['best_match_'+feature_name] = best_match[0]
        else:
            cell['best_match_'+feature_name] = 'none'
    
    # Count cells per best match category
    match_counts = {}
    for ref_name in reference_names + ['none']:
        count = sum(1 for cell in cell_data if cell['best_match_'+feature_name] == ref_name)
        match_counts[ref_name] = count
    
    # Display summary
    print("\nBest match distribution:")
    for ref_name, count in match_counts.items():
        percentage = (count / len(cell_data)) * 100
        print(f"  {ref_name}: {count} cells ({percentage:.1f}%)")
    
    return cell_data

##############################################################################

def temporal_matching(cell_data, Time_stack):
    # Load the reference stack
    print(f"Loading reference stack: {Time_stack}")
    reference_stack = imread(Time_stack)
    z_max, y_max, x_max = reference_stack.shape
    print(f"Reference stack dimensions: z={z_max}, y={y_max}, x={x_max}")

    # Evaluate intensity range
    non_zero_values = reference_stack[reference_stack > 0]
    if len(non_zero_values) > 0:
        min_intensity = np.min(non_zero_values)
        max_intensity = np.max(non_zero_values)
        print(f"Reference stack intensity range: {min_intensity} to {max_intensity}")
    else:
        print("Warning: Reference stack contains no non-zero values")
        return cell_data
    
    # Process each cell
    for cell_idx, cell in enumerate(cell_data):
        if cell_idx % 100 == 0:
            print(f"Processing cell {cell_idx + 1}/{len(cell_data)}")

        # Extract coordinates
        coords = cell['coordinates']

        if isinstance(coords, tuple) and len(coords) == 3:
            # Format: (z_coords, y_coords, x_coords)
            z_coords, y_coords, x_coords = coords
        elif isinstance(coords, list):
            # Format: list of (z, y, x) tuples
            z_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            x_coords = [c[2] for c in coords]
        else:
            print(f"Warning: Unsupported coordinates format for cell {cell_idx}, skipping")
            continue

        # Calculate total number of voxels for this cell
        total_voxels = len(x_coords)

        # Check each coordinate against the reference stack
        matching_intensities = 0

        for i in range(total_voxels):
            # Get coordinates in the correct order
            z, y, x = z_coords[i], y_coords[i], x_coords[i]
            
            intensity = reference_stack[z, y, x]
                
            # Check if this is a positive voxel
            if intensity > 0:
                    matching_intensities += intensity
        
        # Calculate matching statistics
        avg_intensity = (matching_intensities / total_voxels) if total_voxels > 0 else 0
        
        if avg_intensity > 0:
            # Normalize mean intensity to 0-100 range
            norm_intensity = ((avg_intensity - min_intensity) / 
                            (max_intensity - min_intensity) * 100) if max_intensity > min_intensity else 0
            
        else:
            norm_intensity = 0
        
        # Add feature to cell dictionary
        cell[f'Birthdate'] = norm_intensity
    
    return cell_data

##############################################################################

def process_single_cell(cell):
    # Create a copy of the cell dictionary to avoid modifying the original
    result_cell = cell.copy()
    
    # Apply Gaussian smoothing directly
    smoothed_track = gaussian_filter(cell['track'], sigma=SMOOTHING_SIGMA)
    result_cell['track'] = smoothed_track
    
    # Detect spikes
    spike_results = detect_spikes(smoothed_track)
    result_cell.update(spike_results)

    # Create binary track (0 for no spike, 1 for spike)
    binary_track = np.zeros_like(smoothed_track, dtype=int)
    if 'peak_indices' in spike_results and len(spike_results['peak_indices']) > 0:
        binary_track[spike_results['peak_indices']] = 1
    result_cell['binary_track'] = binary_track

    # Extract simplified Fourier features
    fourier_features = extract_features_fourier(smoothed_track)
    result_cell.update(fourier_features)
    
    return result_cell

def process_cells_parallel(cell_data, max_workers=None):

    total_cells = len(cell_data)
    print(f"Processing {total_cells} cells using parallel processing...")
    
    processed_cells = []
    start_time = time.time()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_cell, cell): i 
                            for i, cell in enumerate(cell_data)}
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    processed_cell = future.result()
                    processed_cells.append(processed_cell)
                    
                    # Print progress
                    completed += 1
                    if completed % 100 == 0 or completed == total_cells:
                        print(f"Processed {completed}/{total_cells} cells "
                              f"({completed/total_cells*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error processing cell {idx}: {e}")
                    # Add the original cell to maintain data integrity
                    processed_cells.append(cell_data[idx])
        
        cell_to_idx = {id(cell): idx for idx, cell in enumerate(cell_data)}
        processed_cells.sort(key=lambda x: cell_to_idx.get(id(x), 0))
        
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        processed_cells = process_cells_sequential(cell_data)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return processed_cells

def process_cells_sequential(cell_data):
    processed_cells = []
    total_cells = len(cell_data)
    print(f"Processing {total_cells} cells sequentially...")
    
    start_time = time.time()
    
    for i, cell in enumerate(cell_data):
        if i % 100 == 0 or i == total_cells - 1:
            print(f"Processing cell {i+1}/{total_cells} ({(i+1)/total_cells*100:.1f}%)")
        
        processed_cell = process_single_cell(cell)
        processed_cells.append(processed_cell)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return processed_cells

##############################################################################

def calculate_chunk(chunk_data):
    """Process a chunk of rows for parallel matrix calculation.
    
    Args:
        chunk_data (tuple): (indices, all_tracks) - list of row indices and all tracks data
        
    Returns:
        list: List of (i, j, corr) tuples for calculated correlations
    """
    indices, all_tracks = chunk_data
    results = []
    
    for i in indices:
        track_i = all_tracks[i]
        std_i = np.std(track_i)
        
        for j in range(i, len(all_tracks)):
            track_j = all_tracks[j]
            std_j = np.std(track_j)
            
            # Check if both tracks have variance and sufficient length
            if std_i > 0 and std_j > 0 and len(track_i) > 1 and len(track_j) > 1:
                corr = np.corrcoef(track_i, track_j)[0, 1]
            else:
                corr = 0
                
            results.append((i, j, corr))
    
    return results

def calculate_synchronicity_matrix_parallel(cell_data, use_parallel=True, chunk_size=50):
    n_cells = len(cell_data)
    sync_matrix = np.zeros((n_cells, n_cells))
    
    print(f"Calculating {n_cells}x{n_cells} synchronicity matrix...")
    start_time = time.time()
    
    # Extract all tracks into a matrix for efficient computation
    all_tracks = np.array([cell['track'] for cell in cell_data])
    
    if use_parallel:
        # Create chunks for better parallel performance
        n_chunks = max(1, n_cells // chunk_size)
        chunks = []
        for i in range(0, n_cells, chunk_size):
            end = min(i + chunk_size, n_cells)
            chunks.append((list(range(i, end)), all_tracks))
        
        print(f"  Processing in {len(chunks)} parallel chunks...")
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(calculate_chunk, chunk) for chunk in chunks]
            
            # Track progress
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 1 == 0 or completed == len(chunks):
                    print(f"  Processed {completed}/{len(chunks)} chunks ({completed/len(chunks)*100:.1f}%)")
                
                try:
                    chunk_results = future.result()
                    # Fill in the matrix with the calculated correlations
                    for i, j, corr in chunk_results:
                        sync_matrix[i, j] = corr
                        sync_matrix[j, i] = corr  # Matrix is symmetric
                except Exception as e:
                    print(f"  Error processing chunk: {e}")
    else:
        # Sequential processing
        print("  Using sequential processing...")
        for i in range(n_cells):
            if i % 50 == 0 or i == n_cells-1:
                print(f"  Processing row {i+1}/{n_cells} ({(i+1)/n_cells*100:.1f}%)")
            
            track_i = all_tracks[i]
            std_i = np.std(track_i)
            
            for j in range(i, n_cells):
                track_j = all_tracks[j]
                std_j = np.std(track_j)
                
                # Check if both tracks have variance
                if std_i > 0 and std_j > 0 and len(track_i) > 1 and len(track_j) > 1:
                    corr = np.corrcoef(track_i, track_j)[0, 1]
                else:
                    corr = 0
                    
                sync_matrix[i, j] = corr
                sync_matrix[j, i] = corr  # Matrix is symmetric

    elapsed_time = time.time() - start_time
    print(f"Matrix calculation completed in {elapsed_time:.2f} seconds")
    
    return sync_matrix

def detect_synchronicity_communities(cell_data, sync_matrix=None, use_parallel=True):

    # Calculate synchronicity matrix if not provided
    if sync_matrix is None:
        sync_matrix = calculate_synchronicity_matrix_parallel(cell_data, use_parallel=use_parallel)
    
    # Create a network from the synchronicity matrix
    print("Building network graph...")
    G = nx.Graph()
    n_cells = len(cell_data)
    
    # Add nodes to the graph
    for i in range(n_cells):
        G.add_node(i)
    
    # Add edges to the graph (only for positive correlations)
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            if sync_matrix[i, j] > 0:
                G.add_edge(i, j, weight=sync_matrix[i, j])
    
    # Apply Louvain community detection
    print("Detecting communities...")
    start_time = time.time()
    communities = community_louvain.best_partition(G)
    elapsed_time = time.time() - start_time
    print(f"Community detection completed in {elapsed_time:.2f} seconds")
    
    # Get unique community IDs
    unique_communities = set(communities.values())
    
    # Calculate synchronicity quality for each community
    community_quality = {}
    for comm_id in unique_communities:
        # Get members of this community
        members = [node for node, community in communities.items() if community == comm_id]
        
        if len(members) <= 1:
            # Single-cell communities have no internal synchronicity
            community_quality[comm_id] = 0
            continue
            
        # Calculate average correlation between all pairs within the community
        correlations = []
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                correlations.append(sync_matrix[members[i], members[j]])
        
        # Average correlation as the quality metric
        avg_correlation = np.mean(correlations) if correlations else 0
        community_quality[comm_id] = avg_correlation
    
    # Sort communities by quality (higher mean correlation = better rank)
    sorted_communities = sorted(community_quality.items(), 
                               key=lambda x: x[1], 
                               reverse=True)  # Descending order
    
    # Create mapping from original community ID to new ordered ID (starting from 1)
    community_mapping = {old_id: new_id + 1 for new_id, (old_id, _) in enumerate(sorted_communities)}
    
    # Apply new ordered community labels to cells
    for idx, cell in enumerate(cell_data):
        old_comm_id = communities.get(idx, -1)
        # Map to new ordered ID
        cell['sync_community'] = community_mapping.get(old_comm_id, -1)
    
    # For reporting, count cells per ordered community
    community_counts = {}
    for new_id, (old_id, quality) in enumerate(sorted_communities, 1):
        count = sum(1 for i in range(len(cell_data)) if communities.get(i, -1) == old_id)
        community_counts[new_id] = count
        
    # Display community sizes with quality scores
    print(f"Detected {len(unique_communities)} synchronous communities")
    print("Community sizes (ordered by synchronicity quality):")
    for community_id, count in sorted(community_counts.items()):
        quality_score = sorted_communities[community_id-1][1]  # Get quality for this community
        print(f"  Community {community_id}: {count} cells ({count/len(cell_data)*100:.1f}%) - "
              f"Quality score: {quality_score:.3f}")
    
    return cell_data, sync_matrix, G

##############################################################################

def add_sync_features(cell_data, sync_matrix):
    """Add synchronization features to each cell for PCA analysis"""
    
    # Get community assignments
    communities = [cell.get('sync_community', -1) for cell in cell_data]
    
    # For each cell, calculate sync metrics with each community
    for i, cell in enumerate(cell_data):
        # Overall synchronization strength
        cell['avg_sync_all'] = np.mean(np.abs(sync_matrix[i, :]))
                
        # Calculate within vs between community synchronization
        own_comm = cell.get('sync_community', -1)
        if own_comm != -1:
            # Within community synchronization
            own_indices = [j for j, c in enumerate(communities) if c == own_comm and j != i]
            if own_indices:
                cell['within_comm_sync'] = np.mean(np.abs(sync_matrix[i, own_indices]))
            
            # Between community synchronization
            other_indices = [j for j, c in enumerate(communities) if c != own_comm]
            if other_indices:
                cell['between_comm_sync'] = np.mean(np.abs(sync_matrix[i, other_indices]))
                
        # Calculate sync variability (how consistently the cell co-fires)
        cell['sync_variability'] = np.std(sync_matrix[i, :])
    
    return cell_data

#############################################################################

def calculate_feature_correlation_matrix_parallel(cell_data, features, use_parallel=True, chunk_size=50, max_workers=None):
    """
    Calculate a cell-cell similarity matrix based on feature profiles.
    
    Args:
        cell_data (list): List of cell dictionaries
        features (list): Features to use for similarity calculation
        use_parallel (bool): Whether to use parallel processing
        chunk_size (int): Number of rows to process in each parallel chunk
        max_workers (int): Maximum number of worker processes
        
    Returns:
        numpy.ndarray: Cell-cell feature similarity matrix
    """
    n_cells = len(cell_data)
    feature_sim_matrix = np.zeros((n_cells, n_cells))
    
    print(f"Calculating {n_cells}x{n_cells} feature similarity matrix...")
    start_time = time.time()
    
    # Extract feature vectors for all cells and normalize them
    feature_values = {}
    feature_means = {}
    feature_stds = {}
    
    # Calculate statistics for each feature
    for feature in features:
        values = [cell.get(feature, np.nan) for cell in cell_data]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            feature_means[feature] = np.mean(valid_values)
            feature_stds[feature] = np.std(valid_values)
            if feature_stds[feature] == 0:
                feature_stds[feature] = 1.0  # Avoid division by zero
        else:
            feature_means[feature] = 0.0
            feature_stds[feature] = 1.0
        
        feature_values[feature] = values
    
    # Create normalized feature vectors
    feature_vectors = []
    for i in range(n_cells):
        # Z-score normalize each feature
        normalized_vector = [(feature_values[feature][i] - feature_means[feature]) / feature_stds[feature] 
                            if not np.isnan(feature_values[feature][i]) else np.nan 
                            for feature in features]
        feature_vectors.append(normalized_vector)
    
    # Convert to numpy array for faster processing
    feature_vectors = np.array(feature_vectors)
    
    # Calculate pairwise similarities
    if use_parallel:
        try:
            # Process in parallel chunks
            n_chunks = max(1, n_cells // chunk_size)
            chunks = []
            for i in range(0, n_cells, chunk_size):
                end = min(i + chunk_size, n_cells)
                chunks.append((list(range(i, end)), feature_vectors))
            
            print(f"  Processing in {len(chunks)} parallel chunks...")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_feature_similarity_chunk, chunk) for chunk in chunks]
                
                # Track progress
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed % 10 == 0 or completed == len(chunks):
                        print(f"  Processed {completed}/{len(chunks)} chunks ({completed/len(chunks)*100:.1f}%)")
                    
                    try:
                        chunk_results = future.result()
                        # Fill in the matrix with the calculated similarities
                        for i, j, similarity in chunk_results:
                            feature_sim_matrix[i, j] = similarity
                            feature_sim_matrix[j, i] = similarity  # Matrix is symmetric
                    except Exception as e:
                        print(f"  Error processing chunk: {e}")
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to sequential processing...")
            use_parallel = False
    
    if not use_parallel:
        # Sequential processing
        print("  Using sequential processing...")
        for i in range(n_cells):
            if i % 50 == 0 or i == n_cells-1:
                print(f"  Processing row {i+1}/{n_cells} ({(i+1)/n_cells*100:.1f}%)")
            
            vector_i = feature_vectors[i]
            
            for j in range(i, n_cells):
                i, j, similarity = calculate_cell_feature_similarity(i, j, vector_i, feature_vectors[j])
                feature_sim_matrix[i, j] = similarity
                feature_sim_matrix[j, i] = similarity  # Matrix is symmetric
    
    elapsed_time = time.time() - start_time
    print(f"Feature similarity matrix calculation completed in {elapsed_time:.2f} seconds")
    
    return feature_sim_matrix

def calculate_cell_feature_similarity(i, j, vector_i, vector_j):
    """Calculate similarity between feature vectors of two cells.
    
    Args:
        i (int): First cell index
        j (int): Second cell index
        vector_i (numpy.ndarray): Feature vector for cell i
        vector_j (numpy.ndarray): Feature vector for cell j
        
    Returns:
        tuple: (i, j, similarity value) 
    """
    # Self-similarity is always 1
    if i == j:
        return (i, j, 1.0)
    
    # Find valid features (not NaN in either vector)
    valid_indices = ~np.isnan(vector_i) & ~np.isnan(vector_j)
    
    # Only calculate similarity if we have enough valid features
    if np.sum(valid_indices) >= 2:
        vec1 = vector_i[valid_indices]
        vec2 = vector_j[valid_indices]
        
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((vec1 - vec2)**2))
        
        # Convert distance to similarity (higher is more similar)
        # Scale to range 0-1 with exponential decay
        similarity = np.exp(-distance / np.sqrt(len(vec1)))
    else:
        similarity = 0
    
    return (i, j, similarity)

def process_feature_similarity_chunk(chunk_data):
    """Process a chunk of cell pairs for feature similarity calculation.
    
    Args:
        chunk_data (tuple): (indices, all_vectors)
        
    Returns:
        list: List of (i, j, similarity) tuples
    """
    indices, all_vectors = chunk_data
    results = []
    
    for i in indices:
        vector_i = all_vectors[i]
        
        for j in range(i, len(all_vectors)):
            results.append(calculate_cell_feature_similarity(i, j, vector_i, all_vectors[j]))
    
    return results

def detect_feature_communities(cell_data, feature_sim_matrix, resolution=1.0):
    """
    Detect communities in the feature similarity matrix using Louvain algorithm.
    
    Args:
        cell_data (list): List of cell dictionaries
        feature_sim_matrix (numpy.ndarray): Cell-cell feature similarity matrix
        resolution (float): Resolution parameter for Louvain community detection
        
    Returns:
        tuple: (updated_cell_data, networkx.Graph object)
    """
    n_cells = len(cell_data)
    
    # Create a network from the feature similarity matrix
    print("Building feature network graph...")
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(n_cells):
        G.add_node(i)
    
    # Add edges to the graph (only for positive similarities)
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            if feature_sim_matrix[i, j] > 0:
                G.add_edge(i, j, weight=feature_sim_matrix[i, j])
    
    # Apply Louvain community detection
    print("Detecting feature-based communities...")
    start_time = time.time()
    communities = community_louvain.best_partition(G, resolution=resolution)
    elapsed_time = time.time() - start_time
    print(f"Feature community detection completed in {elapsed_time:.2f} seconds")
    
    # Add community labels to cell data
    for idx, cell in enumerate(cell_data):
        cell['feature_community'] = communities.get(idx, -1) + 1  # Add 1 to start from 1 instead of 0
    
    # Count cells per community
    community_counts = {}
    unique_communities = set(communities.values())
    for community_id in unique_communities:
        count = sum(1 for i in range(n_cells) if communities.get(i, -1) == community_id)
        community_counts[community_id + 1] = count  # Add 1 to match adjusted community IDs
    
    # Display community sizes
    print(f"Detected {len(unique_communities)} feature-based communities")
    print("Feature community sizes:")
    for community_id, count in sorted(community_counts.items()):
        print(f"  Community {community_id}: {count} cells ({count/n_cells*100:.1f}%)")
    
    return cell_data, G

def process_feature_correlation_chunk(chunk_data):
    """Process a chunk of cell pairs for feature similarity.
    
    This function must be at module level to be picklable.
    
    Args:
        chunk_data (tuple): (indices, all_vectors)
        
    Returns:
        list: List of (i, j, similarity) tuples
    """
    indices, all_vectors = chunk_data
    results = []
    
    for i in indices:
        vector_i = all_vectors[i]
        
        for j in range(i, len(all_vectors)):
            results.append(calculate_cell_feature_similarity(i, j, vector_i, all_vectors[j]))
    
    return results

###############################################################

def extract_features(input_path=input_file, output_path=output_file, use_parallel=True, max_workers=None,
                    analyze_synchronicity=True):

    print(f"Loading cell data from {input_path}...")
    cell_data = np.load(input_path, allow_pickle=True)
    print(f"Loaded {len(cell_data)} cells")
    
    # Process cells and extract features
    if use_parallel:
        processed_cells = process_cells_parallel(cell_data, max_workers)
    else:
        processed_cells = process_cells_sequential(cell_data)

    # Apply reference stack evaluation
    if isinstance(Lineage_stacks, dict):
        for feature_name, stack_path in Lineage_stacks.items():
            print(f"\nEvaluating cells against reference stack for feature: {feature_name}")
            processed_cells = spatial_correlation(processed_cells, stack_path, feature_name=feature_name)
        # Determine best match after all references have been evaluated
        processed_cells = determine_best_match(processed_cells, Lineage_stacks, 'Lineage', 5)

    # Add synchronicity analysis if requested
    sync_matrix = None
    if analyze_synchronicity:
        print("\n=== Analyzing cell synchronicity ===")
        processed_cells, sync_matrix, G = detect_synchronicity_communities(processed_cells, use_parallel=use_parallel)

    # Save synchronicity matrix if it was calculated
    if sync_matrix is not None:
        # Generate matrix output path from the main output path
        sync_matrix_path = os.path.join(os.path.dirname(output_path), 'sync_matrix.npy')
        np.save(sync_matrix_path, sync_matrix)
        print(f"Synchronicity matrix saved to {sync_matrix_path}")

    print('Extracting synchronization features')
    processed_cells = add_sync_features(processed_cells, sync_matrix)
    
    # Calculate feature similarity matrix (cell-cell similarity based on features)
    print('\nCalculating feature similarity matrix')
    features_to_compare = ['num_spikes', 'average_spike_width', 'average_spike_amplitude', 
                           'spectral_entropy', 'avg_sync_all']
    feature_sim_matrix = calculate_feature_correlation_matrix_parallel(
        processed_cells, features_to_compare, 
        use_parallel=use_parallel, max_workers=max_workers
    )
    
    # Detect feature-based communities
    print('\nDetecting feature-based communities')
    processed_cells, feature_graph = detect_feature_communities(processed_cells, feature_sim_matrix)
    
    # Save feature similarity matrix
    feature_sim_path = os.path.join(os.path.dirname(output_path), 'feature_similarity_matrix.npy')
    np.save(feature_sim_path, feature_sim_matrix)
    print(f"Feature similarity matrix saved to {feature_sim_path}")
    
    # Save processed data
    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, processed_cells)

    
    return processed_cells

###############################################################

def plot_tracks_spikes(cell_data, output_pdf, cells_per_page=20, max_workers=None):

    # Increase the figure warning threshold for batch processing
    plt.rcParams['figure.max_open_warning'] = 50
    
    num_cells = len(cell_data)
    print(f"Creating PDF visualization for {num_cells} ...")
    
    # Calculate number of pages
    num_pages = (num_cells + cells_per_page - 1) // cells_per_page
    
    # Set default max_workers if None
    if max_workers is None:
        max_workers = os.cpu_count() - 2
    
    # Function to create and immediately save a single page
    def process_page(page_idx):
        start = page_idx * cells_per_page
        end = min((page_idx + 1) * cells_per_page, num_cells)
        
        # Create figure and axes
        fig = plt.figure(figsize=(10, 2 * (end - start)))
        axes = []
        
        # Create subplots
        for i in range(end - start):
            if i == 0:
                ax = fig.add_subplot(end - start, 1, i + 1)
            else:
                ax = fig.add_subplot(end - start, 1, i + 1, sharex=axes[0])
            axes.append(ax)
        
        # Plot each cell on this page
        for i, ax in enumerate(axes):
            cell_idx = start + i
            cell = cell_data[cell_idx]
            
            # Plot the track
            ax.plot(cell['track'], color='blue', linewidth=1)
            
            # Add peak markers if they exist
            if 'peak_indices' in cell and cell['peak_indices'].size > 0:
                peak_indices = cell['peak_indices']
                # Get heights from the track at peak indices
                peak_heights = cell['track'][peak_indices]
                ax.scatter(peak_indices,  peak_heights,
                    color='red', s=20, marker='.')
                num_spikes = len(peak_indices)
            else:
                num_spikes = 0
            
            # Add cell identifier text
            ax.text(0.98, 0.9, f'Cell {cell["cell_id"]} ({num_spikes} spikes)', 
                transform=ax.transAxes, ha='right', fontsize=8)
            
            if i % 5 == 0:
                ax.set_ylabel('Value')
        
        # Add x-label to bottom plot
        axes[-1].set_xlabel('Timepoint')
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, page_idx
    
    # Process the pages in batches to control memory usage
    batch_size = max_workers
    with PdfPages(output_pdf) as pdf:
        for batch_start in range(0, num_pages, batch_size):
            batch_end = min(batch_start + batch_size, num_pages)
            print(f"  Processing pages {batch_start+1}-{batch_end} of {num_pages}...")
            
            # Process this batch of pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_page, page_idx) 
                         for page_idx in range(batch_start, batch_end)]
                
                # Sort completed pages by their index
                completed_pages = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        fig, page_idx = future.result()
                        completed_pages.append((page_idx, fig))
                    except Exception as e:
                        print(f"  Error processing page: {e}")
            
            # Save pages in correct order and immediately close figures
            for page_idx, fig in sorted(completed_pages):
                pdf.savefig(fig)
                plt.close(fig) 

    return None 

if __name__ == '__main__':
    processed_cells=extract_features(use_parallel=True, analyze_synchronicity=True)
    plot_tracks_spikes(processed_cells, output_pdf)
