import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import networkx as nx
import community.community_louvain as community_louvain
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
    
input_cells = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E7/E7_cell_data_features.npy'
input_sync = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E7/sync_matrix.npy'
output_folder = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E7/'

#####################################

def plot_community_matrix(sync_matrix, cell_data, figsize=(12, 12), 
                        title='Cell-Cell Synchronicity Communities'):
    # Extract community assignments
    community_values = [cell.get('sync_community', -1) for cell in cell_data]
    unique_communities = sorted(set(community_values))
    
    # Calculate within-community synchronization for sorting
    community_sync = {}
    for comm_id in unique_communities:
        if comm_id != -1:  # Skip unclassified
            # Get cells in this community
            comm_indices = [i for i, c in enumerate(community_values) if c == comm_id]
            
            if len(comm_indices) > 1:
                # Calculate average synchronization within community
                sync_values = []
                for i, cell_i in enumerate(comm_indices):
                    for j, cell_j in enumerate(comm_indices):
                        if i < j:  # Only include unique pairs
                            sync_values.append(sync_matrix[cell_i, cell_j])
                
                avg_sync = np.mean(sync_values) if sync_values else 0
            else:
                avg_sync = 0
            
            community_sync[comm_id] = avg_sync
    
    # Sort communities by size and synchronization (largest and most synchronized first)
    sorted_communities = sorted(
        [(comm, [i for i, c in enumerate(community_values) if c == comm]) 
         for comm in unique_communities if comm != -1],
        key=lambda x: (community_sync.get(x[0], 0), len(x[1])),
        reverse=True
    )
    
    # Add unclassified at the end if any
    unclassified = [i for i, c in enumerate(community_values) if c == -1]
    if unclassified:
        sorted_communities.append((-1, unclassified))
    
    # Create a new order based on sorted communities
    community_order = []
    community_boundaries = []
    community_info = []
    
    # Create the ordering and track boundaries
    current_idx = 0
    for comm_id, indices in sorted_communities:
        # Add all cells from this community
        community_order.extend(indices)
        
        # Track community size and midpoint for labeling
        size = len(indices)
        mid_point = current_idx + size/2
        community_info.append((mid_point, comm_id, size, community_sync.get(comm_id, 0)))
        
        # Update index and add boundary
        current_idx += size
        if current_idx < len(cell_data):
            community_boundaries.append(current_idx - 0.5)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax_matrix = fig.add_subplot(111)
    
    # Reorder matrix
    ordered_matrix = sync_matrix[np.ix_(community_order, community_order)]
    
    # Plot the matrix with a coolwarm colormap
    im = sns.heatmap(ordered_matrix, ax=ax_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
                    center=0, square=True, xticklabels=False, yticklabels=False,
                    cbar_kws={'shrink': 0.8, 'aspect': 10})
    
    # Add community boundaries with clear black lines
    for boundary in community_boundaries:
        ax_matrix.axhline(boundary, color='black', linewidth=1)
        ax_matrix.axvline(boundary, color='black', linewidth=1)
    
    # Add community labels with size and sync information
    for mid_point, comm_id, size, sync in community_info:
        if comm_id != -1:  # Skip label for unclassified
            ax_matrix.text(mid_point, -0.05, f"C{comm_id} (n={size}, sync={sync:.2f})", 
                        ha='center', va='bottom', fontsize=10, fontweight='normal',
                        color='black', transform=ax_matrix.get_xaxis_transform())
    
    # Set title and adjust layout
    ax_matrix.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig

def plot_community_marker_distribution(cell_data, figsize=(6, 8), Group='Lineage', title='Marker Distribution by Community', bar_width=0.8):
    """
    Create a stacked bar chart showing the relative distribution of markers within each community.
    Each bar represents 100% of cells within that community, with segments showing marker proportions.
    Added bar_width parameter to control spacing between bars.
    """
    # Get community labels
    communities = [cell.get('sync_community', -1) for cell in cell_data]
    unique_communities = sorted(set(communities))
    
    # Get marker labels
    markers = [cell.get('best_match_'+Group, 'none') for cell in cell_data]
    unique_markers = sorted(set(markers))

    def marker_order_key(marker):
        # Define priority for marker ordering from bottom to top of stacked bar
        marker_lower = marker.lower()
        if 'none' in marker_lower:
            return 3  # 'none' markers at bottom
        elif 'glut' in marker_lower:
            return 2  # 'glut' markers in middle
        elif 'gad' in marker_lower:
            return 1  # 'gad' markers at top
        else:
            return 0  # Other markers above these
    
    # Sort markers by custom order
    unique_markers = sorted(unique_markers, key=marker_order_key)
    
    # Generate colors for markers
    cmap = plt.cm.get_cmap('Set3', len(unique_markers))
    marker_colors = {marker: cmap(i) for i, marker in enumerate(unique_markers)}
    if 'none' in marker_colors:
        marker_colors['none'] = '#949494'
    
    # Count cells for each community-marker combination
    count_data = []
    for comm in unique_communities:
        community_cells = [i for i, c in enumerate(communities) if c == comm]
        community_size = len(community_cells)
        
        for marker in unique_markers:
            # Count cells with this marker in this community
            count = sum(1 for i in community_cells if markers[i] == marker)
            percentage = (count / community_size * 100) if community_size > 0 else 0
            
            count_data.append({
                'Community': f'Community {comm}',
                'Marker': marker,
                'Count': count,
                'Percentage': percentage
            })
    
    # Create DataFrame
    df = pd.DataFrame(count_data)
    
    # Pivot to get the right format for a stacked bar chart
    pivot_df = df.pivot(index='Community', columns='Marker', values='Percentage')
    pivot_df = pivot_df.fillna(0)  # Replace NaN with 0
    pivot_df = pivot_df[unique_markers]
    # Also create a dataframe with the actual counts for annotations
    count_pivot = df.pivot(index='Community', columns='Marker', values='Count')
    count_pivot = count_pivot.fillna(0)  # Replace NaN with 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # CHANGE: Use matplotlib's bar function directly instead of pandas plot
    # This allows us to control the bar width more precisely
    x = np.arange(len(pivot_df.index))
    bottom = np.zeros(len(pivot_df.index))
    
    for marker in unique_markers:
        values = pivot_df[marker].values
        ax.bar(x, values, bottom=bottom, width=bar_width, 
               label=marker, color=marker_colors[marker])
        bottom += values
    
    # Add labels
    ax.set_xlabel('Community')
    ax.set_ylabel('Percentage of Cells (%)')
    ax.set_title(title)
    ax.set_ylim(0, 100)  # Set y-axis to 0-100%
    
    # Set x-tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Marker', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate X-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add annotations with actual cell counts
    for i, comm_label in enumerate(pivot_df.index):
        # For each community, calculate the positions of annotations
        cumulative_percentage = 0
        
        for marker in unique_markers:
            # Get percentage and count for this marker in this community
            if marker in pivot_df.columns:
                percentage = pivot_df.loc[comm_label, marker]
                count = int(count_pivot.loc[comm_label, marker])
                
                # Only annotate if segment is large enough
                if count > 0:
                    # Position annotation in the middle of this segment
                    y_pos = cumulative_percentage + (percentage / 2)
                    
                    # Add annotation with count
                    ax.text(i, y_pos, str(count),
                           ha='center', va='center',
                           fontsize=8, color='white', fontweight='bold')
                
                # Update cumulative percentage for next marker
                cumulative_percentage += percentage
    
    # Tighten layout to make better use of space
    plt.tight_layout()
    return fig

#####################################

def plot_community_average_tracks(cell_data, figsize=(15, 10), title='Average Cell Tracks by Community'):

    # Get community labels
    labels = [cell.get('sync_community', -1) for cell in cell_data]
    unique_labels = sorted(set(labels))
    
    # Create a figure with subplots
    n_communities = len(unique_labels)
    n_cols = min(3, n_communities)
    n_rows = (n_communities + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=figsize)
    
    # Create a GridSpec with extra space for titles
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Calculate and plot average track for each community
    for i, label in enumerate(unique_labels):
        # Create subplot
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get cells in this community
        community_cells = [cell for cell, l in zip(cell_data, labels) if l == label]
        community_size = len(community_cells)
        
        if community_size == 0:
            continue
        
        # Extract tracks and calculate average
        # First, we need to ensure all tracks have the same length
        track_length = len(community_cells[0]['track'])
        valid_tracks = [cell['track'] for cell in community_cells 
                       if len(cell['track']) == track_length]
        
        
        if len(valid_tracks) > 0:
            # Calculate average track
            avg_track = np.mean(valid_tracks, axis=0)
            
            # Calculate standard deviation
            std_track = np.std(valid_tracks, axis=0)
            
            # Time points for x-axis
            x = np.arange(len(avg_track)) / 2.0
            
            # Plot average track
            ax.plot(x, avg_track, 'b-', linewidth=1)
            
            # Plot standard deviation area
            upper_bound = avg_track + std_track
            
            ax.fill_between(x, avg_track, upper_bound, 
                          color='b', alpha=0.3)
            
            # Add community info
            ax.set_title(f'Community {label} (n={community_size})')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Signal')
            
            # Ensure y-axis doesn't go below zero
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, f'No valid tracks for\nCommunity {label}',
                  ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(title)
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    return fig

def plot_combined_community_analysis(cell_data, sync_matrix, output_folder, 
                                 markers_per_page=4, Group='Lineage'):
    """
    Creates a comprehensive analysis figure for each marker showing:
    a) Marker average signal
    b) Community analysis directly on all cells of a specific marker (no filtering)
    c) Community synchronicity matrix
    d) Feature difference
    e) Birthdate distribution by community within marker (NEW)
    f) Spatial scattering within communities (NEW)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Set all cells to be part of a single marker
    markers = ["All Cells"] * len(cell_data)
    unique_markers = ["All Cells"]  # Just one marker for all cells

    # Sort markers to place those containing "Gad" or "gad" first
    def marker_sort_key(marker):
        if "Gad" in marker or "gad" in marker:
            return (0, marker)  # Tuple with 0 as first element for Gad markers
        return (1, marker)      # Tuple with 1 as first element for non-Gad markers
    
    unique_markers.sort(key=marker_sort_key)
    
    # Get numerical features for feature difference analysis, excluding specific features
    exclude_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 'sync_community', 
                        'feature_community', 'within_comm_sync', 'between_comm_sync', 'sync_variability', 'frequency_avg']
    exclude_features = ['Birthdate', 'spatial_scatter'] 
    numerical_features = []
    for key in cell_data[0].keys():
        if any(key.startswith(prefix) for prefix in exclude_prefixes) or '_match' in key or key in exclude_features:
            continue
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    # Pre-normalize all numerical features to 0-1 scale across all cells
    normalized_features = {}
    for feature in numerical_features:
        values = [cell.get(feature, 0) for cell in cell_data]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1.0  # Avoid division by zero
        
        # Store normalized values by cell index
        for i, val in enumerate(values):
            if i not in normalized_features:
                normalized_features[i] = {}
            normalized_features[i][feature] = (val - min_val) / range_val
    
    # Precompute cell centroids for spatial proximity calculations
    print("Precomputing cell centroids for spatial analysis...")
    cell_centroids = {}
    for i, cell in enumerate(cell_data):
        if 'coordinates' in cell:
            coords = cell['coordinates']
            try:
                # Handle coordinate format (z_coords, y_coords, x_coords)
                if isinstance(coords, tuple) and len(coords) == 3:
                    z_coords, y_coords, x_coords = coords
                    # Calculate centroid as mean position
                    centroid = np.array([
                        np.mean(z_coords),
                        np.mean(y_coords),
                        np.mean(x_coords)
                    ])
                    cell_centroids[i] = centroid
            except Exception as e:
                print(f"  Error calculating centroid for cell {i}: {e}")
    
    # Determine global y-axis range for all track plots
    print("Calculating global y-axis range for consistent plotting...")
    all_valid_tracks = []
    for m_idx, marker in enumerate(unique_markers):
        marker_cells_indices = [i for i, m in enumerate(markers) if m == marker]
        for i in marker_cells_indices:
            if 'track' in cell_data[i] and len(cell_data[i]['track']) > 0:
                all_valid_tracks.append(cell_data[i]['track'])
    
    global_y_min = 0  # Keep 0 as minimum
    if all_valid_tracks:
        all_track_values = np.concatenate(all_valid_tracks)
        global_y_max = np.percentile(all_track_values, 99.5)  # Use 99.5th percentile to avoid outliers
        print(f"Global y-axis range: [0, {global_y_max:.2f}]")
    else:
        global_y_max = 1.0
        print("Warning: No valid tracks found. Using default y-axis range [0, 1.0]")
    
    # Define pastel color palette that reflects standard community colors
    community_colors = [
        '#9ecae1',  # Pastel blue (C1)
        '#ffeda0',  # Pastel yellow (C2)
        '#a1d99b',  # Pastel green (C3)
        '#fc9272'   # Pastel red (C4)
    ]
    
    # Calculate number of pages needed
    n_pages = (len(unique_markers) + markers_per_page - 1) // markers_per_page
    
    saved_figures = []
    
    # Process each page
    for page in range(n_pages):
        # Get markers for this page
        page_markers = unique_markers[page * markers_per_page:(page + 1) * markers_per_page]
        n_markers_on_page = len(page_markers)
        
        # Adjust figure width for two additional plots (10 columns total now)
        fig = plt.figure(figsize=(45, 6 * n_markers_on_page))
        
        # Process each marker on this page
        for m_idx, marker in enumerate(page_markers):
            print(f"Processing {marker} ({m_idx+1}/{len(page_markers)})")
            
            # Get cells with this marker
            marker_cells_indices = [i for i, m in enumerate(markers) if m == marker]
            
            if len(marker_cells_indices) < 5:  # Need at least 5 cells for meaningful analysis
                print(f"  Not enough cells for {marker}, skipping")
                continue
                
            # Extract tracks
            valid_tracks = []
            valid_indices = []
            for i in marker_cells_indices:
                if 'track' in cell_data[i] and len(cell_data[i]['track']) > 0:
                    valid_tracks.append(cell_data[i]['track'])
                    valid_indices.append(i)
            
            if not valid_tracks:
                print(f"  No valid tracks for {marker}, skipping")
                continue
                
            # Time points for x-axis (assuming 2Hz sampling)
            x = np.arange(len(valid_tracks[0])) / 2.0
            
            # Use global y-axis range
            y_min = global_y_min
            y_max = global_y_max
            
            # 1. Plot average marker signal (column 0)
            ax_avg = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 0), colspan=1)
            avg_track = np.mean(valid_tracks, axis=0)
            std_track = np.std(valid_tracks, axis=0)
            
            ax_avg.plot(x, avg_track, 'b-', linewidth=1.5)
            ax_avg.fill_between(x, avg_track - std_track, avg_track + std_track, 
                              alpha=0.3, color='b')
            ax_avg.set_title(f'{marker} Average (n={len(valid_tracks)})')
            ax_avg.set_xlabel('Time (s)')
            ax_avg.set_ylabel('Signal')
            ax_avg.set_ylim(y_min, y_max)  # Use global y-axis
            
            # 2. Apply Louvain community detection on ALL cells of this marker
            if len(valid_indices) >= 3:  # Need at least 3 cells for meaningful communities
                # Extract synchronization submatrix for marker cells
                sync_submatrix = sync_matrix[np.ix_(valid_indices, valid_indices)]
                
                # Create a graph (only include positive correlations)
                G = nx.Graph()
                for i in range(len(valid_indices)):
                    G.add_node(i)  # Add node for each cell
                    
                # Add edges with positive weights (we only care about positive synchronization)
                for i in range(len(valid_indices)):
                    for j in range(i+1, len(valid_indices)):
                        weight = sync_submatrix[i, j]
                        if weight > 0:  # Only include positive correlations
                            G.add_edge(i, j, weight=weight)
                
                # Apply Louvain community detection 
                try:
                    # Try the most common import pattern
                    import community.community_louvain as community_louvain
                    partition = community_louvain.best_partition(G)
                except (ImportError, AttributeError):
                    try:
                        # Try direct import
                        import community
                        partition = community.best_partition(G)
                    except (ImportError, AttributeError):
                        try:
                            # Another common pattern
                            from community import community_louvain
                            partition = community_louvain.best_partition(G)
                        except (ImportError, AttributeError):
                            try:
                                # Last attempt - direct import of function
                                from community import best_partition
                                partition = best_partition(G)
                            except (ImportError, AttributeError):
                                print("ERROR: Could not import community detection module.")
                                print("Please install with: pip install python-louvain")
                                continue
                
                # Extract communities
                community_ids = set(partition.values())
                n_communities = len(community_ids)
                
                # Create a mapping from community ID to list of cell indices
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(valid_indices[node])
                
                # Calculate average synchronization within each community
                community_sync = {}
                for comm_id, cell_indices in communities.items():
                    if len(cell_indices) > 1:
                        sync_values = []
                        for i, cell_i in enumerate(cell_indices):
                            for j, cell_j in enumerate(cell_indices):
                                if i < j:  # Only include unique pairs
                                    sync_values.append(sync_matrix[cell_i, cell_j])
                        
                        avg_sync = np.mean(sync_values) if sync_values else 0
                    else:
                        avg_sync = 0
                    community_sync[comm_id] = avg_sync
                
                # Calculate spatial scattering for cells in each community
                community_spatial_scatter = {}
                for comm_id, cell_indices in communities.items():
                    if len(cell_indices) > 1:
                        # Get centroids for cells in this community
                        community_centroids = [
                            cell_centroids[i] for i in cell_indices 
                            if i in cell_centroids
                        ]
                        
                        if len(community_centroids) > 1:
                            # Calculate pairwise distances between centroids
                            total_distance = 0
                            pair_count = 0
                            
                            for i in range(len(community_centroids)):
                                for j in range(i+1, len(community_centroids)):
                                    try:
                                        dist = np.linalg.norm(community_centroids[i] - community_centroids[j])
                                        total_distance += dist
                                        pair_count += 1
                                    except Exception as e:
                                        pass
                            
                            # Calculate average distance
                            avg_distance = total_distance / pair_count if pair_count > 0 else 0
                            community_spatial_scatter[comm_id] = avg_distance
                            
                            # Add this as a normalized feature for each cell in the community
                            for idx in cell_indices:
                                if idx not in normalized_features:
                                    normalized_features[idx] = {}
                                # More scattered = higher value, so use the raw distance
                                normalized_features[idx]['spatial_scatter'] = min(1.0, avg_distance / 100.0)  # Normalize to 0-1
                        else:
                            community_spatial_scatter[comm_id] = 0
                    else:
                        community_spatial_scatter[comm_id] = 0
                
                # Sort communities by size and synchronization (largest and most synchronized first)
                sorted_communities = sorted(
                    communities.items(), 
                    key=lambda x: (community_sync[x[0]], len(x[1])), 
                    reverse=True
                )
                
                # If we have many communities, limit to the top 4
                max_communities = min(4, len(sorted_communities))
                top_communities = sorted_communities[:max_communities]
                
                # Plot each community in its own column
                for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                    # Plot in a dedicated column
                    ax_comm = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 1 + c_idx), colspan=1)
                    
                    # Get tracks for cells in this community
                    comm_tracks = [cell_data[i]['track'] for i in cell_indices 
                                 if 'track' in cell_data[i]]
                    
                    if comm_tracks:
                        # Calculate average track
                        avg_track = np.mean(comm_tracks, axis=0)
                        std_track = np.std(comm_tracks, axis=0)
                        
                        # Plot with community color
                        color = community_colors[c_idx % len(community_colors)]
                        ax_comm.plot(x, avg_track, '-', linewidth=1.5, color=color)
                        ax_comm.fill_between(x, avg_track - std_track, avg_track + std_track, 
                                           alpha=0.3, color=color)
                        
                        # Only show community number and sync in title
                        ax_comm.set_title(f'Community {c_idx+1} (n={len(comm_tracks)})\nSync: {community_sync[comm_id]:.2f}')
                        
                        ax_comm.set_xlabel('Time (s)')
                        ax_comm.set_ylabel('Signal')
                        ax_comm.set_ylim(y_min, y_max)  # Use global y-axis
                    else:
                        ax_comm.text(0.5, 0.5, 'No tracks', ha='center', va='center', 
                                   transform=ax_comm.transAxes)
                               
                # Feature diff with narrower width and community-matched colors
                ax_feat = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 5), colspan=2)
                
                # Select top 5 features for visualization
                n_top_features = min(4, len(numerical_features))
                
                # Extract pre-normalized feature values for each community
                feature_values_by_comm = [[] for _ in range(max_communities)]
                
                for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                    for idx in cell_indices:
                        # Use normalized features
                        if idx in normalized_features:
                            features = [normalized_features[idx].get(f, 0) for f in numerical_features]
                            feature_values_by_comm[c_idx].append(features)
                
                # Calculate mean feature values for each community
                comm_means = []
                for comm_features in feature_values_by_comm:
                    if comm_features:  # Check if not empty
                        comm_means.append(np.mean(comm_features, axis=0))
                    else:
                        comm_means.append(np.zeros(len(numerical_features)))
                        
                comm_means = np.array(comm_means)
                
                # Select top features based on variance between communities
                feature_variance = np.var(comm_means, axis=0)
                top_indices = np.argsort(feature_variance)[-n_top_features:]
                
                # Sort top indices by descending variance for display
                sorted_idx = sorted(range(len(top_indices)), 
                                   key=lambda i: feature_variance[top_indices[i]], 
                                   reverse=True)
                top_indices = [top_indices[i] for i in sorted_idx]
                
                # Prepare data for plotting
                top_features = [numerical_features[i] for i in top_indices]
                top_importance = comm_means[:, top_indices]
                
                # Create bar plots with matching community colors
                bar_width = 0.8 / max_communities
                for c_idx in range(min(max_communities, len(top_communities))):
                    x_pos = np.arange(n_top_features) + c_idx * bar_width - (max_communities-1) * bar_width/2
                    color = community_colors[c_idx % len(community_colors)]
                    ax_feat.bar(x_pos, top_importance[c_idx], width=bar_width, 
                              label=f'Comm {c_idx+1}', color=color)
                
                ax_feat.set_xticks(np.arange(n_top_features))
                ax_feat.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
                ax_feat.set_title('Feature differences')
                ax_feat.set_ylabel('Normalized Value')
                ax_feat.set_ylim(0, 1)  # Enforce 0-1 scale
                ax_feat.legend(fontsize='small', loc='upper right')
                
                # Birthdate distribution by community
                ax_birthdate = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 7), colspan=1)
                
                # Check if 'birthdate' exists in the data
                if any('Birthdate' in cell for cell in cell_data):
                    # Get birthdate values for each community
                    birthdates_by_comm = []
                    community_labels = []
                    
                    for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                        # Extract birthdates for this community
                        comm_birthdates = [cell_data[i].get('Birthdate', np.nan) for i in cell_indices]
                        comm_birthdates = [bd for bd in comm_birthdates if not np.isnan(bd)]
                        
                        if comm_birthdates:
                            birthdates_by_comm.append(comm_birthdates)
                            community_labels.append(f'C{c_idx+1}')
                    
                    if birthdates_by_comm:
                        # Create violin or box plot
                        violin_parts = ax_birthdate.violinplot(birthdates_by_comm, showmeans=True)
                        
                        # Color the violins
                        for i, pc in enumerate(violin_parts['bodies']):
                            pc.set_facecolor(community_colors[i % len(community_colors)])
                            pc.set_alpha(0.7)
                        
                        # Set x-ticks and labels
                        ax_birthdate.set_xticks(np.arange(1, len(community_labels) + 1))
                        ax_birthdate.set_xticklabels(community_labels)
                        ax_birthdate.set_title('Birthdate Distribution')
                        ax_birthdate.set_ylabel('Birthdate')
                    else:
                        ax_birthdate.text(0.5, 0.5, 'No birthdate data',
                                       ha='center', va='center', transform=ax_birthdate.transAxes)
                else:
                    ax_birthdate.text(0.5, 0.5, 'Birthdate feature\nnot found',
                                   ha='center', va='center', transform=ax_birthdate.transAxes)
                
                # Spatial scattering within communities
                ax_spatial = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 8), colspan=1)
                
                # Plot spatial scatter values for each community
                comm_ids = []
                scatter_values = []
                
                for c_idx, (comm_id, _) in enumerate(top_communities):
                    if comm_id in community_spatial_scatter:
                        comm_ids.append(f'C{c_idx+1}')
                        scatter_values.append(community_spatial_scatter[comm_id])
                
                if scatter_values:
                    # Create bar plot
                    bars = ax_spatial.bar(np.arange(len(scatter_values)), scatter_values, 
                                      color=[community_colors[i % len(community_colors)] for i in range(len(scatter_values))])
                    
                    # Add values above bars
                    for i, v in enumerate(scatter_values):
                        ax_spatial.text(i, v * 0.75, f'{v:.1f}', 
                                    ha='center', va='center', fontsize=8,
                                    color='white', fontweight='bold')
                    
                    # Set x-ticks and labels
                    ax_spatial.set_xticks(np.arange(len(comm_ids)))
                    ax_spatial.set_xticklabels(comm_ids)
                    ax_spatial.set_title('Spatial Scatter')
                    ax_spatial.set_ylabel('Avg Distance (pixels)')
                else:
                    ax_spatial.text(0.5, 0.5, 'No spatial data',
                                 ha='center', va='center', transform=ax_spatial.transAxes)
            else:
                # Not enough cells for meaningful Louvain detection
                for c_idx in range(4):  # 4 community columns
                    ax_comm = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 1 + c_idx), colspan=1)
                    ax_comm.text(0.5, 0.5, f'Not enough cells\nfor clustering\n(need at least 3)', 
                               ha='center', va='center', transform=ax_comm.transAxes)
                    ax_comm.set_ylim(y_min, y_max)  # Keep consistent y-axis even for empty plots
                
                
                ax_feat = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 5), colspan=2)
                ax_feat.text(0.5, 0.5, 'Not enough cells\nfor feature analysis', 
                           ha='center', va='center', transform=ax_feat.transAxes)
                ax_feat.set_ylim(0, 1)  # Keep consistent y-axis
                
                # Empty birthdate and spatial plots
                ax_birthdate = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 7), colspan=1)
                ax_birthdate.text(0.5, 0.5, 'Not enough cells\nfor birthdate analysis', 
                               ha='center', va='center', transform=ax_birthdate.transAxes)
                
                ax_spatial = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 8), colspan=1)
                ax_spatial.text(0.5, 0.5, 'Not enough cells\nfor spatial analysis', 
                             ha='center', va='center', transform=ax_spatial.transAxes)
        
        # Add overall title for the page
        plt.suptitle(f'{Group} Marker Analysis (Page {page+1}/{n_pages})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Save the figure
        output_path = os.path.join(output_folder, f'{Group}_combined_analysis_page{page+1}.tiff')
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        
        saved_figures.append(output_path)
        print(f"Saved page {page+1} to {output_path}")
    
    return saved_figures

def plot_combined_marker_analysis(cell_data, sync_matrix, output_folder, 
                                 markers_per_page=4, Group='Lineage'):
    """
    Creates a comprehensive analysis figure for each marker showing:
    a) Marker average signal
    b) Community analysis directly on all cells of a specific marker (no filtering)
    c) Community synchronicity matrix
    d) Feature difference
    e) Birthdate distribution by community within marker (NEW)
    f) Spatial scattering within communities (NEW)
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get marker labels
    markers = [cell.get(f'best_match_{Group}', 'none') for cell in cell_data]
    unique_markers = sorted(set([m for m in markers if m != 'none']))
    
    # Sort markers to place those containing "Gad" or "gad" first
    def marker_sort_key(marker):
        if "Gad" in marker or "gad" in marker:
            return (0, marker)  # Tuple with 0 as first element for Gad markers
        return (1, marker)      # Tuple with 1 as first element for non-Gad markers
    
    unique_markers.sort(key=marker_sort_key)
    
    # Get numerical features for feature difference analysis, excluding specific features
    exclude_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 'sync_community', 
                        'feature_community', 'within_comm_sync', 'between_comm_sync', 'sync_variability', 'frequency_avg']
    exclude_features = ['Birthdate', 'spatial_scatter'] 
    numerical_features = []
    for key in cell_data[0].keys():
        if any(key.startswith(prefix) for prefix in exclude_prefixes) or '_match' in key or key in exclude_features:
            continue
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    # Pre-normalize all numerical features to 0-1 scale across all cells
    normalized_features = {}
    for feature in numerical_features:
        values = [cell.get(feature, 0) for cell in cell_data]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1.0  # Avoid division by zero
        
        # Store normalized values by cell index
        for i, val in enumerate(values):
            if i not in normalized_features:
                normalized_features[i] = {}
            normalized_features[i][feature] = (val - min_val) / range_val
    
    # Precompute cell centroids for spatial proximity calculations
    print("Precomputing cell centroids for spatial analysis...")
    cell_centroids = {}
    for i, cell in enumerate(cell_data):
        if 'coordinates' in cell:
            coords = cell['coordinates']
            try:
                # Handle coordinate format (z_coords, y_coords, x_coords)
                if isinstance(coords, tuple) and len(coords) == 3:
                    z_coords, y_coords, x_coords = coords
                    # Calculate centroid as mean position
                    centroid = np.array([
                        np.mean(z_coords),
                        np.mean(y_coords),
                        np.mean(x_coords)
                    ])
                    cell_centroids[i] = centroid
            except Exception as e:
                print(f"  Error calculating centroid for cell {i}: {e}")
    
    # Determine global y-axis range for all track plots
    print("Calculating global y-axis range for consistent plotting...")
    all_valid_tracks = []
    for m_idx, marker in enumerate(unique_markers):
        marker_cells_indices = [i for i, m in enumerate(markers) if m == marker]
        for i in marker_cells_indices:
            if 'track' in cell_data[i] and len(cell_data[i]['track']) > 0:
                all_valid_tracks.append(cell_data[i]['track'])
    
    global_y_min = 0  # Keep 0 as minimum
    if all_valid_tracks:
        all_track_values = np.concatenate(all_valid_tracks)
        global_y_max = np.percentile(all_track_values, 99.5)  # Use 99.5th percentile to avoid outliers
        print(f"Global y-axis range: [0, {global_y_max:.2f}]")
    else:
        global_y_max = 1.0
        print("Warning: No valid tracks found. Using default y-axis range [0, 1.0]")
    
    # Define pastel color palette that reflects standard community colors
    community_colors = [
        '#9ecae1',  # Pastel blue (C1)
        '#ffeda0',  # Pastel yellow (C2)
        '#a1d99b',  # Pastel green (C3)
        '#fc9272'   # Pastel red (C4)
    ]
    
    # Calculate number of pages needed
    n_pages = (len(unique_markers) + markers_per_page - 1) // markers_per_page
    
    saved_figures = []
    
    # Process each page
    for page in range(n_pages):
        # Get markers for this page
        page_markers = unique_markers[page * markers_per_page:(page + 1) * markers_per_page]
        n_markers_on_page = len(page_markers)
        
        # Adjust figure width for two additional plots (10 columns total now)
        fig = plt.figure(figsize=(45, 6 * n_markers_on_page))
        
        # Process each marker on this page
        for m_idx, marker in enumerate(page_markers):
            print(f"Processing {marker} ({m_idx+1}/{len(page_markers)})")
            
            # Get cells with this marker
            marker_cells_indices = [i for i, m in enumerate(markers) if m == marker]
            
            if len(marker_cells_indices) < 5:  # Need at least 5 cells for meaningful analysis
                print(f"  Not enough cells for {marker}, skipping")
                continue
                
            # Extract tracks
            valid_tracks = []
            valid_indices = []
            for i in marker_cells_indices:
                if 'track' in cell_data[i] and len(cell_data[i]['track']) > 0:
                    valid_tracks.append(cell_data[i]['track'])
                    valid_indices.append(i)
            
            if not valid_tracks:
                print(f"  No valid tracks for {marker}, skipping")
                continue
                
            # Time points for x-axis (assuming 2Hz sampling)
            x = np.arange(len(valid_tracks[0])) / 2.0
            
            # Use global y-axis range
            y_min = global_y_min
            y_max = global_y_max
            
            # 1. Plot average marker signal (column 0)
            ax_avg = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 0), colspan=1)
            avg_track = np.mean(valid_tracks, axis=0)
            std_track = np.std(valid_tracks, axis=0)
            
            ax_avg.plot(x, avg_track, 'b-', linewidth=1.5)
            ax_avg.fill_between(x, avg_track - std_track, avg_track + std_track, 
                              alpha=0.3, color='b')
            ax_avg.set_title(f'{marker} Average (n={len(valid_tracks)})')
            ax_avg.set_xlabel('Time (s)')
            ax_avg.set_ylabel('Signal')
            ax_avg.set_ylim(y_min, y_max)  # Use global y-axis
            
            # 2. Apply Louvain community detection on ALL cells of this marker
            if len(valid_indices) >= 3:  # Need at least 3 cells for meaningful communities
                # Extract synchronization submatrix for marker cells
                sync_submatrix = sync_matrix[np.ix_(valid_indices, valid_indices)]
                
                # Create a graph (only include positive correlations)
                G = nx.Graph()
                for i in range(len(valid_indices)):
                    G.add_node(i)  # Add node for each cell
                    
                # Add edges with positive weights (we only care about positive synchronization)
                for i in range(len(valid_indices)):
                    for j in range(i+1, len(valid_indices)):
                        weight = sync_submatrix[i, j]
                        if weight > 0:  # Only include positive correlations
                            G.add_edge(i, j, weight=weight)
                
                # Apply Louvain community detection 
                try:
                    # Try the most common import pattern
                    import community.community_louvain as community_louvain
                    partition = community_louvain.best_partition(G)
                except (ImportError, AttributeError):
                    try:
                        # Try direct import
                        import community
                        partition = community.best_partition(G)
                    except (ImportError, AttributeError):
                        try:
                            # Another common pattern
                            from community import community_louvain
                            partition = community_louvain.best_partition(G)
                        except (ImportError, AttributeError):
                            try:
                                # Last attempt - direct import of function
                                from community import best_partition
                                partition = best_partition(G)
                            except (ImportError, AttributeError):
                                print("ERROR: Could not import community detection module.")
                                print("Please install with: pip install python-louvain")
                                continue
                
                # Extract communities
                community_ids = set(partition.values())
                n_communities = len(community_ids)
                
                # Create a mapping from community ID to list of cell indices
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(valid_indices[node])
                
                # Calculate average synchronization within each community
                community_sync = {}
                for comm_id, cell_indices in communities.items():
                    if len(cell_indices) > 1:
                        sync_values = []
                        for i, cell_i in enumerate(cell_indices):
                            for j, cell_j in enumerate(cell_indices):
                                if i < j:  # Only include unique pairs
                                    sync_values.append(sync_matrix[cell_i, cell_j])
                        
                        avg_sync = np.mean(sync_values) if sync_values else 0
                    else:
                        avg_sync = 0
                    community_sync[comm_id] = avg_sync
                
                # Calculate spatial scattering for cells in each community
                community_spatial_scatter = {}
                for comm_id, cell_indices in communities.items():
                    if len(cell_indices) > 1:
                        # Get centroids for cells in this community
                        community_centroids = [
                            cell_centroids[i] for i in cell_indices 
                            if i in cell_centroids
                        ]
                        
                        if len(community_centroids) > 1:
                            # Calculate pairwise distances between centroids
                            total_distance = 0
                            pair_count = 0
                            
                            for i in range(len(community_centroids)):
                                for j in range(i+1, len(community_centroids)):
                                    try:
                                        dist = np.linalg.norm(community_centroids[i] - community_centroids[j])
                                        total_distance += dist
                                        pair_count += 1
                                    except Exception as e:
                                        pass
                            
                            # Calculate average distance
                            avg_distance = total_distance / pair_count if pair_count > 0 else 0
                            community_spatial_scatter[comm_id] = avg_distance
                            
                            # Add this as a normalized feature for each cell in the community
                            for idx in cell_indices:
                                if idx not in normalized_features:
                                    normalized_features[idx] = {}
                                # More scattered = higher value, so use the raw distance
                                normalized_features[idx]['spatial_scatter'] = min(1.0, avg_distance / 100.0)  # Normalize to 0-1
                        else:
                            community_spatial_scatter[comm_id] = 0
                    else:
                        community_spatial_scatter[comm_id] = 0
                
                # Sort communities by size and synchronization (largest and most synchronized first)
                sorted_communities = sorted(
                    communities.items(), 
                    key=lambda x: (community_sync[x[0]], len(x[1])), 
                    reverse=True
                )
                
                # If we have many communities, limit to the top 4
                max_communities = min(4, len(sorted_communities))
                top_communities = sorted_communities[:max_communities]
                
                # Plot each community in its own column
                for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                    # Plot in a dedicated column
                    ax_comm = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 1 + c_idx), colspan=1)
                    
                    # Get tracks for cells in this community
                    comm_tracks = [cell_data[i]['track'] for i in cell_indices 
                                 if 'track' in cell_data[i]]
                    
                    if comm_tracks:
                        # Calculate average track
                        avg_track = np.mean(comm_tracks, axis=0)
                        std_track = np.std(comm_tracks, axis=0)
                        
                        # Plot with community color
                        color = community_colors[c_idx % len(community_colors)]
                        ax_comm.plot(x, avg_track, '-', linewidth=1.5, color=color)
                        ax_comm.fill_between(x, avg_track - std_track, avg_track + std_track, 
                                           alpha=0.3, color=color)
                        
                        # Only show community number and sync in title
                        ax_comm.set_title(f'Community {c_idx+1} (n={len(comm_tracks)})\nSync: {community_sync[comm_id]:.2f}')
                        
                        ax_comm.set_xlabel('Time (s)')
                        ax_comm.set_ylabel('Signal')
                        ax_comm.set_ylim(y_min, y_max)  # Use global y-axis
                    else:
                        ax_comm.text(0.5, 0.5, 'No tracks', ha='center', va='center', 
                                   transform=ax_comm.transAxes)
                
                # Plot synchronicity matrix for communities
                ax_sync = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 5), colspan=1)
                
                # Collect indices from all communities
                all_community_indices = []
                for _, cell_indices in top_communities:
                    all_community_indices.extend(cell_indices)
                
                if all_community_indices:
                    # Extract and reorder the submatrix
                    ordered_submatrix = sync_matrix[np.ix_(all_community_indices, all_community_indices)]
                    
                    # Plot the matrix WITHOUT colorbar
                    im = ax_sync.imshow(ordered_submatrix, cmap='coolwarm', vmin=-1, vmax=1)
                    
                    # Add community boundaries
                    boundaries = []
                    count = 0
                    for _, cell_indices in top_communities:
                        count += len(cell_indices)
                        if count < len(all_community_indices):
                            boundaries.append(count - 0.5)
                    
                    for boundary in boundaries:
                        ax_sync.axhline(boundary, color='black', linewidth=1)
                        ax_sync.axvline(boundary, color='black', linewidth=1)
                        
                    ax_sync.set_title(f'{marker} Sync Matrix')
                    ax_sync.set_xticks([])
                    ax_sync.set_yticks([])
                else:
                    ax_sync.text(0.5, 0.5, 'No communities\nfor sync matrix', 
                               ha='center', va='center', transform=ax_sync.transAxes)
                
                # Feature diff with narrower width and community-matched colors
                ax_feat = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 6), colspan=2)
                
                # Select top 5 features for visualization
                n_top_features = min(4, len(numerical_features))
                
                # Extract pre-normalized feature values for each community
                feature_values_by_comm = [[] for _ in range(max_communities)]
                
                for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                    for idx in cell_indices:
                        # Use normalized features
                        if idx in normalized_features:
                            features = [normalized_features[idx].get(f, 0) for f in numerical_features]
                            feature_values_by_comm[c_idx].append(features)
                
                # Calculate mean feature values for each community
                comm_means = []
                for comm_features in feature_values_by_comm:
                    if comm_features:  # Check if not empty
                        comm_means.append(np.mean(comm_features, axis=0))
                    else:
                        comm_means.append(np.zeros(len(numerical_features)))
                        
                comm_means = np.array(comm_means)
                
                # Select top features based on variance between communities
                feature_variance = np.var(comm_means, axis=0)
                top_indices = np.argsort(feature_variance)[-n_top_features:]
                
                # Sort top indices by descending variance for display
                sorted_idx = sorted(range(len(top_indices)), 
                                   key=lambda i: feature_variance[top_indices[i]], 
                                   reverse=True)
                top_indices = [top_indices[i] for i in sorted_idx]
                
                # Prepare data for plotting
                top_features = [numerical_features[i] for i in top_indices]
                top_importance = comm_means[:, top_indices]
                
                # Create bar plots with matching community colors
                bar_width = 0.8 / max_communities
                for c_idx in range(min(max_communities, len(top_communities))):
                    x_pos = np.arange(n_top_features) + c_idx * bar_width - (max_communities-1) * bar_width/2
                    color = community_colors[c_idx % len(community_colors)]
                    ax_feat.bar(x_pos, top_importance[c_idx], width=bar_width, 
                              label=f'Comm {c_idx+1}', color=color)
                
                ax_feat.set_xticks(np.arange(n_top_features))
                ax_feat.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
                ax_feat.set_title('Feature differences')
                ax_feat.set_ylabel('Normalized Value')
                ax_feat.set_ylim(0, 1)  # Enforce 0-1 scale
                ax_feat.legend(fontsize='small', loc='upper right')
                
                # Birthdate distribution by community
                ax_birthdate = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 8), colspan=1)
                
                # Check if 'birthdate' exists in the data
                if any('Birthdate' in cell for cell in cell_data):
                    # Get birthdate values for each community
                    birthdates_by_comm = []
                    community_labels = []
                    
                    for c_idx, (comm_id, cell_indices) in enumerate(top_communities):
                        # Extract birthdates for this community
                        comm_birthdates = [cell_data[i].get('Birthdate', np.nan) for i in cell_indices]
                        comm_birthdates = [bd for bd in comm_birthdates if not np.isnan(bd)]
                        
                        if comm_birthdates:
                            birthdates_by_comm.append(comm_birthdates)
                            community_labels.append(f'C{c_idx+1}')
                    
                    if birthdates_by_comm:
                        # Create violin or box plot
                        violin_parts = ax_birthdate.violinplot(birthdates_by_comm, showmeans=True)
                        
                        # Color the violins
                        for i, pc in enumerate(violin_parts['bodies']):
                            pc.set_facecolor(community_colors[i % len(community_colors)])
                            pc.set_alpha(0.7)
                        
                        # Set x-ticks and labels
                        ax_birthdate.set_xticks(np.arange(1, len(community_labels) + 1))
                        ax_birthdate.set_xticklabels(community_labels)
                        ax_birthdate.set_title('Birthdate Distribution')
                        ax_birthdate.set_ylabel('Birthdate')
                    else:
                        ax_birthdate.text(0.5, 0.5, 'No birthdate data',
                                       ha='center', va='center', transform=ax_birthdate.transAxes)
                else:
                    ax_birthdate.text(0.5, 0.5, 'Birthdate feature\nnot found',
                                   ha='center', va='center', transform=ax_birthdate.transAxes)
                
                # Spatial scattering within communities
                ax_spatial = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 9), colspan=1)
                
                # Plot spatial scatter values for each community
                comm_ids = []
                scatter_values = []
                
                for c_idx, (comm_id, _) in enumerate(top_communities):
                    if comm_id in community_spatial_scatter:
                        comm_ids.append(f'C{c_idx+1}')
                        scatter_values.append(community_spatial_scatter[comm_id])
                
                if scatter_values:
                    # Create bar plot
                    bars = ax_spatial.bar(np.arange(len(scatter_values)), scatter_values, 
                                      color=[community_colors[i % len(community_colors)] for i in range(len(scatter_values))])
                    
                    # Add values above bars
                    for i, v in enumerate(scatter_values):
                        ax_spatial.text(i, v * 0.75, f'{v:.1f}', 
                                    ha='center', va='center', fontsize=8,
                                    color='white', fontweight='bold')
                    
                    # Set x-ticks and labels
                    ax_spatial.set_xticks(np.arange(len(comm_ids)))
                    ax_spatial.set_xticklabels(comm_ids)
                    ax_spatial.set_title('Spatial Scatter')
                    ax_spatial.set_ylabel('Avg Distance (pixels)')
                else:
                    ax_spatial.text(0.5, 0.5, 'No spatial data',
                                 ha='center', va='center', transform=ax_spatial.transAxes)
            else:
                # Not enough cells for meaningful Louvain detection
                for c_idx in range(4):  # 4 community columns
                    ax_comm = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 1 + c_idx), colspan=1)
                    ax_comm.text(0.5, 0.5, f'Not enough cells\nfor clustering\n(need at least 3)', 
                               ha='center', va='center', transform=ax_comm.transAxes)
                    ax_comm.set_ylim(y_min, y_max)  # Keep consistent y-axis even for empty plots
                
                # Matrix and feature diff
                ax_sync = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 5), colspan=1)
                ax_sync.text(0.5, 0.5, 'Not enough cells\nfor sync matrix', 
                           ha='center', va='center', transform=ax_sync.transAxes)
                
                ax_feat = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 6), colspan=2)
                ax_feat.text(0.5, 0.5, 'Not enough cells\nfor feature analysis', 
                           ha='center', va='center', transform=ax_feat.transAxes)
                ax_feat.set_ylim(0, 1)  # Keep consistent y-axis
                
                # Empty birthdate and spatial plots
                ax_birthdate = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 8), colspan=1)
                ax_birthdate.text(0.5, 0.5, 'Not enough cells\nfor birthdate analysis', 
                               ha='center', va='center', transform=ax_birthdate.transAxes)
                
                ax_spatial = plt.subplot2grid((n_markers_on_page, 10), (m_idx, 9), colspan=1)
                ax_spatial.text(0.5, 0.5, 'Not enough cells\nfor spatial analysis', 
                             ha='center', va='center', transform=ax_spatial.transAxes)
        
        # Add overall title for the page
        plt.suptitle(f'{Group} Marker Analysis (Page {page+1}/{n_pages})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Save the figure
        output_path = os.path.join(output_folder, f'{Group}_combined_analysis_page{page+1}.tiff')
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        
        saved_figures.append(output_path)
        print(f"Saved page {page+1} to {output_path}")
    
    return saved_figures

#####################################

def get_numerical_features(cell_data):
    """Helper function to get numerical features, excluding specific keys"""
    numerical_features = []
    for key in cell_data[0].keys():
        # Check if the value is numerical
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    return numerical_features

#####################################

def process_cluster(args):
    """Process a single cluster to calculate feature statistics.
    
    Args:
        args (tuple): Contains (level, label, cell_data, numerical_features)
        
    Returns:
        tuple: (level, label, feature_stats)
    """
    level, label, cell_data, numerical_features = args
    
    # Get cluster labels
    labels = [cell.get(level, -1) for cell in cell_data]
    
    # Get cells in this cluster
    cluster_cells = [cell for cell, l in zip(cell_data, labels) if l == label]
    
    # Calculate feature statistics
    feature_stats = {}
    for feature in numerical_features:
        values = [cell.get(feature, 0) for cell in cluster_cells]
        feature_stats[feature] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return (level, label, feature_stats)

def analyze_clusters(cell_data, output_folder, use_parallel=True, max_workers=None):
    """Generate detailed analysis and statistics focused on sync communities."""
    print("Starting community analysis...")
    start_time = time.time()
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Focus only on 'sync_community'
    level_keys = ['sync_community']
    
    # Identify numerical features
    exclude_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 
                     'sync_community']
    
    numerical_features = []
    for key in cell_data[0].keys():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    # Generate statistics for each level
    level_stats = {}
    
    for level in level_keys:  # This will only process 'sync_community'
        print(f"Processing level: {level}")
        level_start = time.time()
        
        # Get cluster labels
        labels = [cell.get(level, -1) for cell in cell_data]
        unique_labels = sorted(set(labels))
        
        # Calculate basic statistics
        n_clusters = len(unique_labels)
        cluster_sizes = {label: labels.count(label) for label in unique_labels}
        
        # Create tasks for parallel processing
        cluster_tasks = [(level, label, cell_data, numerical_features) for label in unique_labels]
        
        # Process clusters in parallel if requested
        if use_parallel and len(unique_labels) > 1:
            print(f"  Processing {len(unique_labels)} communities in parallel...")
            cluster_features = {}
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all cluster tasks
                futures = [executor.submit(process_cluster, task) for task in cluster_tasks]
                
                # Track progress
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    if completed % 100 == 0 or completed == n_clusters:
                        print(f"  Processed {completed}/{n_clusters} communities ({completed/n_clusters*100:.1f}%)")
                    
                    try:
                        level, label, stats = future.result()
                        cluster_features[label] = stats
                    except Exception as e:
                        print(f"  Error processing community: {e}")
            
            # Extract features for the current level
            level_cluster_features = cluster_features
        else:
            print(f"  Processing {len(unique_labels)} communities sequentially...")
            level_cluster_features = {}
            for task in cluster_tasks:
                level, label, stats = process_cluster(task)
                level_cluster_features[label] = stats
        
        # Store level statistics
        level_stats[level] = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'cluster_features': level_cluster_features
        }
        
        level_elapsed = time.time() - level_start
        print(f"Completed level {level} in {level_elapsed:.2f} seconds")
    
    # Save statistics to CSV files
    print("Saving results to CSV files...")
    
    # Create CSV files for sync_community only
    for level, stats in level_stats.items():
        # Create a DataFrame for community sizes
        df_sizes = pd.DataFrame({
            'Community': list(stats['cluster_sizes'].keys()),
            'Size': list(stats['cluster_sizes'].values()),
            'Percentage': [100 * size / len(cell_data) for size in stats['cluster_sizes'].values()]
        })
        
        # Save community sizes
        sizes_path = os.path.join(output_folder, f'{level}_community_sizes.csv')
        df_sizes.to_csv(sizes_path, index=False)
        
        # Save feature statistics for each community
        for cluster, features in stats['cluster_features'].items():
            # Create a DataFrame for feature statistics
            df_features = pd.DataFrame({
                'Feature': list(features.keys()),
                'Mean': [stat['mean'] for stat in features.values()],
                'Std': [stat['std'] for stat in features.values()],
                'Min': [stat['min'] for stat in features.values()],
                'Max': [stat['max'] for stat in features.values()],
                'Median': [stat['median'] for stat in features.values()]
            })
            
            # Save feature statistics
            features_path = os.path.join(output_folder, f'{level}_community_{cluster}_features.csv')
            df_features.to_csv(features_path, index=False)
    
    total_elapsed = time.time() - start_time
    print(f"Analysis completed in {total_elapsed:.2f} seconds")
    
    # Return analysis results
    results = {
        'level_stats': level_stats
    }
    
    return results

def analyze_markers(cell_data, output_folder):
    """Generate detailed analysis and statistics focused on marker groups."""
    print("Starting marker analysis...")
    start_time = time.time()
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    marker_groups = ['Lineage', 'Dependancy']
    
    for group in marker_groups:
        print(f"Processing marker group: {group}")
        
        # Get marker labels
        markers = [cell.get(f'best_match_{group}', 'none') for cell in cell_data]
        unique_markers = sorted(set(markers))
        
        # Calculate basic statistics
        marker_sizes = {marker: markers.count(marker) for marker in unique_markers}
        
        # Get numerical features
        exclude_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 'sync_community'] 
        numerical_features = []
        for key in cell_data[0].keys():
            if any(key.startswith(prefix) for prefix in exclude_prefixes):
                continue
            
            value = cell_data[0].get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numerical_features.append(key)
        
        # Create Excel file for the marker group
        excel_path = os.path.join(output_folder, f'{group}_marker_features.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            # Create marker size sheet
            sizes_df = pd.DataFrame({
                'Marker': list(marker_sizes.keys()),
                'Size': list(marker_sizes.values()),
                'Percentage': [100 * size / len(cell_data) for size in marker_sizes.values()]
            })
            sizes_df.to_excel(writer, sheet_name='Marker_Sizes', index=False)
            
            # For each feature, create a sheet with marker statistics
            for feature in numerical_features:
                # Gather statistics for each marker
                feature_stats = {}
                feature_stats['Marker'] = []
                feature_stats['Mean'] = []
                feature_stats['StdDev'] = []
                feature_stats['Min'] = []
                feature_stats['Max'] = []
                feature_stats['Median'] = []
                
                for marker in unique_markers:
                    # Get cells with this marker
                    marker_cells = [cell for i, cell in enumerate(cell_data) if markers[i] == marker]
                    
                    # Extract feature values
                    values = [cell.get(feature, 0) for cell in marker_cells]
                    
                    # Calculate statistics
                    feature_stats['Marker'].append(marker)
                    feature_stats['Mean'].append(np.mean(values))
                    feature_stats['StdDev'].append(np.std(values))
                    feature_stats['Min'].append(np.min(values))
                    feature_stats['Max'].append(np.max(values))
                    feature_stats['Median'].append(np.median(values))
                
                # Create DataFrame for this feature
                feature_df = pd.DataFrame(feature_stats)
                
                # Excel has a 31 character limit for sheet names
                sheet_name = feature[:31]
                feature_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Saved {group} marker statistics to {excel_path}")
        
        # Create a raw data file for each marker
        raw_excel_path = os.path.join(output_folder, f'{group}_marker_raw_data.xlsx')
        with pd.ExcelWriter(raw_excel_path) as writer:
            for marker in unique_markers:
                # Get cells with this marker
                marker_cells = [cell for i, cell in enumerate(cell_data) if markers[i] == marker]
                
                if not marker_cells:
                    continue
                
                # Create a DataFrame with all features for these cells
                raw_data = {feature: [cell.get(feature, 0) for cell in marker_cells] 
                          for feature in numerical_features}
                
                # Add cell_id if available
                if 'cell_id' in cell_data[0]:
                    raw_data['cell_id'] = [cell.get('cell_id', i) for i, cell in enumerate(marker_cells)]
                
                raw_df = pd.DataFrame(raw_data)
                
                # Excel has a 31 character limit for sheet names
                sheet_name = marker[:31]
                raw_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Saved {group} raw data to {raw_excel_path}")
    
    total_elapsed = time.time() - start_time
    print(f"Marker analysis completed in {total_elapsed:.2f} seconds")

#####################################

def create_analysis_report(output_folder, cell_data, sync_matrix):
    saved_files = []
    
    # Plot 1: Community Matrix
    fig = plot_community_matrix(sync_matrix, cell_data,
                             title='Cell-Cell Synchronicity by Community')
    output_path = os.path.join(output_folder, "01_community_matrix.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    # Community-Marker distribution
    fig = plot_community_marker_distribution(cell_data)
    output_path = os.path.join(output_folder, "01b_marker_distribution_by_community.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    fig = plot_community_marker_distribution(cell_data, Group='Dependancy')
    output_path = os.path.join(output_folder, "01c_Dependancy_distribution_by_community.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    # Use community analysis with the same layout as marker analysis
    output_path = os.path.join(output_folder, 'community_analysis')
    os.makedirs(output_path, exist_ok=True)
    
    # Generate analysis for Lineage markers in communities
    community_files = plot_combined_community_analysis(
            cell_data=cell_data,
            sync_matrix=sync_matrix,
            output_folder=output_path,
            markers_per_page=4,
        Group='Lineage'
    )
    saved_files.extend(community_files)
    
    # Keep the marker analysis
    for group in ['Lineage', 'Dependancy']:
        output_path = os.path.join(output_folder, 'combined_analysis')
        os.makedirs(output_path, exist_ok=True)
        
        plot_combined_marker_analysis(
            cell_data=cell_data,
            sync_matrix=sync_matrix,
            output_folder=output_path,
            markers_per_page=4,
            Group=group
        )

    print(f"Analysis images saved to: {output_folder}")
    return saved_files

if __name__ == '__main__': 
    # Performance settings
    use_parallel = True   # Set to True to use parallel processing
    max_workers = None    # None will use all available CPU cores
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading data from {input_cells}...")
    start_time = time.time()
    
    # Load data
    cell_data = np.load(input_cells, allow_pickle=True)
    sync_matrix = np.load(input_sync)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds. Found {len(cell_data)} cells.")
    
    # Create analysis folder
    analysis_folder = os.path.join(output_folder, 'analysis')
    os.makedirs(analysis_folder, exist_ok=True)

    # Generate cluster analysis focused on communities
    analyze_clusters(cell_data, analysis_folder, use_parallel=use_parallel, max_workers=max_workers)
    analyze_markers(cell_data, analysis_folder)    
    # Create images folder for TIFF files
    images_folder = os.path.join(output_folder, 'figures')
    os.makedirs(images_folder, exist_ok=True)
    
    # Create individual TIFF images instead of PDF report
    print("Generating TIFF images...")
    images_start = time.time()
    saved_images = create_analysis_report(images_folder, cell_data, sync_matrix)
    images_time = time.time() - images_start
    print(f"TIFF images generated in {images_time:.2f} seconds")
    print(f"Saved {len(saved_images)} images in {images_folder}")
    
    total_time = time.time() - start_time
    print(f"Total processing completed in {total_time:.2f} seconds")