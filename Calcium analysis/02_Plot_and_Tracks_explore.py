import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
import time
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
    
input_cells = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1_cell_data_features.npy'
input_sync = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/sync_matrix.npy'
output_folder = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/'

#####################################

def plot_community_matrix(sync_matrix, cell_data, figsize=(12, 12), 
                        title='Cell-Cell Synchronicity Communities'):
    
    # Extract community assignments
    community_values = [cell.get('sync_community', -1) for cell in cell_data]
    unique_communities = sorted(set(community_values))
        
    # Create an order that groups cells by community
    community_order = []
    for comm in unique_communities:
        if comm != -1:  # Skip unclassified
            # Add all cells from this community
            community_order.extend([i for i, c in enumerate(community_values) if c == comm])
    
    # Add unclassified at the end if any
    community_order.extend([i for i, c in enumerate(community_values) if c == -1])
    
    # Create figure with grid for precise layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(100, 100, figure=fig)

    # Main matrix plot
    ax_matrix = fig.add_subplot() 
    
    # Reorder matrix
    ordered_matrix = sync_matrix[community_order, :][:, community_order]
    
    # Plot matrix with smaller colorbar
    sns.heatmap(ordered_matrix, ax=ax_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
              center=0, square=True, xticklabels=False, yticklabels=False,
              cbar_kws={'shrink': 0.8, 'aspect': 10})  # Make colorbar smaller
    
    # Store boundary positions for ticks
    community_boundaries = []
    community_info = []
    
    # Draw community indicators
    ordered_communities = [community_values[i] for i in community_order]
    current_comm = ordered_communities[0]
    start_idx = 0
    
    for i, comm in enumerate(ordered_communities):
        if comm != current_comm or i == len(ordered_communities) - 1:
            # We've reached a new community or the end
            if i == len(ordered_communities) - 1 and comm == current_comm:
                end_idx = i + 1  # Include the last element
            else:
                end_idx = i
                community_boundaries.append(i)
                
            # Store community label info
            community_size = end_idx - start_idx
            mid_point = start_idx + community_size/2
            community_info.append((mid_point, current_comm, community_size))
            
            # Update for next community
            current_comm = comm
            start_idx = i
    
    # Add community labels below the matrix
    for mid_point, comm, size in community_info:
        if size > len(ordered_communities) / 20:  # Skip very small communities
            ax_matrix.text(mid_point, -0.05, f"C{comm} (n={size})", 
                        ha='center', va='bottom', fontsize=10, fontweight='normal',
                        color='black', transform=ax_matrix.get_xaxis_transform())
    
    # Add tick marks at community boundaries
    ax_matrix.set_xticks(community_boundaries)
    ax_matrix.set_xticklabels([])  # No labels for ticks
    ax_matrix.tick_params(axis='x', length=6, width=1, which='major', bottom=True, top=False)
        
    return fig

def plot_community_marker_distribution(cell_data, figsize=(12, 8), Group='Lineage', title='Marker Distribution by Community'):
    """
    Create a stacked bar chart showing the relative distribution of markers within each community.
    Each bar represents 100% of cells within that community, with segments showing marker proportions.
    """
    # Get community labels
    communities = [cell.get('sync_community', -1) for cell in cell_data]
    unique_communities = sorted(set(communities))
    
    # Get marker labels
    markers = [cell.get('best_match_'+Group, 'none') for cell in cell_data]
    unique_markers = sorted(set(markers))
    
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
    
    # Also create a dataframe with the actual counts for annotations
    count_pivot = df.pivot(index='Community', columns='Marker', values='Count')
    count_pivot = count_pivot.fillna(0)  # Replace NaN with 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bar chart
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=[marker_colors[m] for m in pivot_df.columns])
    
    # Add labels
    ax.set_xlabel('Community')
    ax.set_ylabel('Percentage of Cells (%)')
    ax.set_title(title)
    ax.set_ylim(0, 100)  # Set y-axis to 0-100%
    
    # Add legend
    ax.legend(title='Marker', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rotate X-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add annotations with actual cell counts - completely rewritten approach
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

def plot_marker_average_tracks(cell_data, figsize=(15, 10), Group='Lineage', title='Average Cell Tracks by Marker'):
    # Get best match labels
    best_matches = [cell.get('best_match_'+Group, 'none') for cell in cell_data]
    unique_matches = sorted(set(best_matches))
    
    # Create a figure with subplots
    n_matches = len(unique_matches)
    n_cols = min(3, n_matches)
    n_rows = (n_matches + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=figsize)
    
    # Create a GridSpec with extra space for titles
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Calculate and plot average track for each match type
    for i, match_type in enumerate(unique_matches):
        # Create subplot
        row, col = i // n_cols, i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get cells with this match type
        match_cells = [cell for cell, m in zip(cell_data, best_matches) if m == match_type]
        match_size = len(match_cells)
        
        if match_size == 0:
            continue
        
        # Extract tracks and calculate average
        # First, ensure all tracks have the same length
        track_length = len(match_cells[0]['track'])
        valid_tracks = [cell['track'] for cell in match_cells 
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
            ax.fill_between(x, avg_track - std_track, avg_track + std_track, color='b', alpha=0.3)
            # Add match type info
            ax.set_title(f'{match_type} (n={match_size})')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Signal')
            # Ensure y-axis doesn't go below zero
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, f'No valid tracks for\n{match_type}',
                  ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(title)
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    return fig

def plot_marker_activity_patterns(cell_data, figsize=(18, 12), Group='Lineage', n_patterns=3):
    """
    Identify different activity patterns within each marker type by clustering
    track time series, then analyze which features are most associated with each pattern.
    """
    # Get marker labels
    markers = [cell.get('best_match_' + Group, 'none') for cell in cell_data]
    unique_markers = sorted(set(markers))
    
    # Get numerical features for later analysis (excluding tracks and spatial features)
    exclude_prefixes = ['track', 'coordinates', 'cell_id', 'peak_indices', 'sync_community', 'feature_community', 'between_comm_sync', 'within_comm_sync']
    numerical_features = []
    
    for key in cell_data[0].keys():
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Skip features related to spatial correlation (ending with _match)
            if not key.endswith('_match'):
                numerical_features.append(key)
    
    # Create figure
    n_markers = len(unique_markers)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_markers, n_patterns + 1, figure=fig, hspace=1, wspace=0.4,
                 width_ratios=[1] * n_patterns + [0.6])
    
    for m_idx, marker in enumerate(unique_markers):
        # Get cells with this marker
        marker_cells = []
        marker_indices = []
        
        for i, cell in enumerate(cell_data):
            if markers[i] == marker:
                marker_cells.append(cell)
                marker_indices.append(i)
        
        if len(marker_cells) < n_patterns + 1:
            # Skip if too few cells
            continue
        
        # Extract tracks for clustering
        valid_tracks = []
        valid_indices = []
        
        for i, cell in enumerate(marker_cells):
            if 'track' in cell and len(cell['track']) > 0:
                # Ensure all tracks have the same length
                valid_tracks.append(cell['track'])
                valid_indices.append(marker_indices[i])
        
        if len(valid_tracks) < n_patterns:
            # Handle case with insufficient valid data
            continue
            
        # Convert to numpy array for clustering
        X_tracks = np.array(valid_tracks)
        
        # Standardize tracks for clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Reshape for StandardScaler (samples, features)
            X_reshaped = X_tracks.reshape(X_tracks.shape[0], -1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reshaped)
            # Reshape back to original shape for easier handling
            X_scaled = X_scaled.reshape(X_tracks.shape)
        
        # Apply K-means clustering on the track patterns
        # Need to reshape to 2D for K-means
        kmeans = KMeans(n_clusters=n_patterns, random_state=42, n_init=10)
        pattern_labels = kmeans.fit_predict(X_reshaped)
        
        # Extract numerical features for each cluster
        feature_values_by_pattern = [[] for _ in range(n_patterns)]
        
        for i, idx in enumerate(valid_indices):
            pattern = pattern_labels[i]
            cell = cell_data[idx]
            features = [cell.get(f, 0) for f in numerical_features]
            feature_values_by_pattern[pattern].append(features)
        
        # Calculate feature importance as normalized difference between clusters
        feature_importance = np.zeros((n_patterns, len(numerical_features)))
        
        # Calculate mean feature values for each pattern
        pattern_means = []
        for pattern_features in feature_values_by_pattern:
            if pattern_features:  # Check if not empty
                pattern_means.append(np.mean(pattern_features, axis=0))
            else:
                pattern_means.append(np.zeros(len(numerical_features)))
                
        pattern_means = np.array(pattern_means)
        
        # Normalize feature values across patterns
        feature_ranges = np.ptp(pattern_means, axis=0)
        feature_ranges[feature_ranges == 0] = 1  # Avoid division by zero
        normalized_means = pattern_means / feature_ranges[None, :]
        
        # Use normalized means as feature importance
        feature_importance = np.abs(normalized_means)
        
        # Plot each pattern
        for p_idx in range(n_patterns):
            ax = fig.add_subplot(gs[m_idx, p_idx])
            
            # Get tracks for this pattern
            pattern_track_indices = [i for i, label in enumerate(pattern_labels) if label == p_idx]
            
            if not pattern_track_indices:
                ax.text(0.5, 0.5, 'No cells', ha='center', va='center', 
                       transform=ax.transAxes)
                continue
            
            # Extract tracks for this pattern
            pattern_tracks = [valid_tracks[i] for i in pattern_track_indices]
            
            if not pattern_tracks:
                continue
                
            # Calculate average track
            avg_track = np.mean(pattern_tracks, axis=0)
            std_track = np.std(pattern_tracks, axis=0)
            
            # Time points
            x = np.arange(len(avg_track)) / 2.0
            
            # Plot
            ax.plot(x, avg_track, '-', linewidth=1.5)
            ax.fill_between(x, avg_track - std_track, avg_track + std_track, 
                          alpha=0.3)
            
            ax.set_title(f'{marker} - Pattern {p_idx+1} (n={len(pattern_tracks)})')
            if p_idx == 0:
                ax.set_ylabel('Signal')
            ax.set_xlabel('Time (s)')
            ax.set_ylim(bottom=0)  # Set Y-axis minimum to 0
            
        # Plot feature importance for each pattern
        ax_feat = fig.add_subplot(gs[m_idx, n_patterns])
        
        # Select top features for visualization
        n_top_features = min(8, len(numerical_features))
        importance_sum = np.sum(feature_importance, axis=0)
        top_indices = np.argsort(importance_sum)[-n_top_features:]

        # Sort top indices by descending importance for display
        sorted_idx = sorted(range(len(top_indices)), key=lambda i: importance_sum[top_indices[i]], reverse=True)
        top_indices = [top_indices[i] for i in sorted_idx]

        # Prepare data for plotting
        top_features = [numerical_features[i] for i in top_indices]
        top_importance = feature_importance[:, top_indices]
        
        # Create bar plots
        bar_width = 0.8 / n_patterns
        for p_idx in range(n_patterns):
            x_pos = np.arange(n_top_features) + p_idx * bar_width - (n_patterns-1) * bar_width/2
            ax_feat.bar(x_pos, top_importance[p_idx], width=bar_width, 
                      label=f'Pattern {p_idx+1}')
        
        ax_feat.set_xticks(np.arange(n_top_features))
        ax_feat.set_xticklabels(top_features, rotation=45, ha='right')
        ax_feat.set_title('Feature Importance')
        ax_feat.set_ylabel('Normalized Feature Expression')
        ax_feat.legend(fontsize='small')
    
    plt.suptitle(f'Activity Patterns by {Group}')
    plt.tight_layout()
    return fig

def plot_marker_sync_patterns(cell_data, sync_matrix, figsize=(20, 15), Group='Lineage', n_clusters=3):
    """
    Plot average tracks for markers, with cells grouped by their synchronization patterns
    using clustering to find actual synchronized cell groups.
    """
    # Get marker labels
    markers = [cell.get('best_match_' + Group, 'none') for cell in cell_data]
    unique_markers = sorted(set(markers))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    n_markers = len(unique_markers)
    n_cols = n_clusters
    n_rows = n_markers
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, hspace=0.4, wspace=0.3,
                  width_ratios=[1] * n_cols + [0.3])
    
    from sklearn.cluster import KMeans
    
    for m_idx, marker in enumerate(unique_markers):
        # Get cells with this marker
        marker_cell_indices = [i for i, m in enumerate(markers) if m == marker]
        
        if len(marker_cell_indices) < n_clusters + 1:  # Need at least n_clusters+1 cells
            continue
            
        # Extract synchronization submatrix for these cells
        sync_submatrix = sync_matrix[np.ix_(marker_cell_indices, marker_cell_indices)]
        
        # Use K-means clustering on the sync matrix to find synchronized clusters
        # First reshape the submatrix to be suitable for clustering
        # We'll use each cell's synchronization profile with other cells as its feature vector
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sync_submatrix)
        
        # Create clusters based on K-means results
        clusters = []
        for cluster_id in range(n_clusters):
            # Get cells in this cluster
            cluster_indices = [marker_cell_indices[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # Calculate average within-cluster synchronization
            if len(cluster_indices) > 1:
                within_sync_values = []
                for i, cell_i in enumerate(cluster_indices):
                    for j, cell_j in enumerate(cluster_indices):
                        if i < j:  # Only include unique pairs
                            within_sync_values.append(sync_matrix[cell_i, cell_j])
                
                avg_sync = np.mean(within_sync_values) if within_sync_values else 0
            else:
                avg_sync = 0
                
            clusters.append((cluster_indices, avg_sync))
        
        # Sort clusters by their internal synchronization (highest first)
        clusters.sort(key=lambda x: x[1], reverse=True)
        
        # Plot each cluster
        for c_idx, (cell_indices, avg_sync) in enumerate(clusters):
            ax = fig.add_subplot(gs[m_idx, c_idx])
            
            # Get tracks for cells in this cluster
            cluster_tracks = [cell_data[i]['track'] for i in cell_indices if 'track' in cell_data[i]]
            
            if not cluster_tracks:
                ax.text(0.5, 0.5, 'No tracks available', ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Calculate average track
            avg_track = np.mean(cluster_tracks, axis=0)
            std_track = np.std(cluster_tracks, axis=0)
            
            # Time points for x-axis
            x = np.arange(len(avg_track)) / 2.0  # Assuming 2Hz sampling
            
            # Plot average track
            ax.plot(x, avg_track, 'b-', linewidth=1)
            ax.fill_between(x, avg_track - std_track, avg_track + std_track, 
                          color='b', alpha=0.3)
            
            # Add title with sync info
            sync_level = "Cluster " + str(c_idx + 1)
            ax.set_title(f'{marker} - {sync_level} (n={len(cluster_tracks)})\nAvg Sync: {avg_sync:.2f}')
            
            if c_idx == 0:
                ax.set_ylabel('Signal')
            ax.set_xlabel('Time (s)')
            ax.set_ylim(bottom=0)
        
        # Add sync matrix visualization for this marker
        ax_sync = fig.add_subplot(gs[m_idx, n_clusters])
        
        # Reorder cells by cluster for better visualization
        cluster_order = []
        for cluster_indices, _ in clusters:
            cluster_order.extend(cluster_indices)
        
        if cluster_order:
            # Extract and reorder the submatrix
            ordered_submatrix = sync_matrix[np.ix_(cluster_order, cluster_order)]
            im = ax_sync.imshow(ordered_submatrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add cluster boundaries
            boundaries = []
            count = 0
            for cell_indices, _ in clusters:
                count += len(cell_indices)
                if count < len(cluster_order):
                    boundaries.append(count - 0.5)
            
            for boundary in boundaries:
                ax_sync.axhline(boundary, color='black', linewidth=1)
                ax_sync.axvline(boundary, color='black', linewidth=1)
                
            ax_sync.set_title(f'{marker}\nSync Matrix')
        else:
            ax_sync.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_sync.transAxes)
    
    plt.suptitle(f'Signal Patterns by {Group} - Grouped by Synchronization Clusters')
    return fig

def plot_intensity_split_by_marker(cell_data, figsize=(15, 15), threshold=500, Group='Lineage'):
    """
    Plot signal tracks split by intensity threshold for each marker/community.
    """
    # Get marker labels
    markers = [cell.get('best_match_' + Group, 'none') for cell in cell_data]
    unique_markers = sorted(set(markers))
    
    # Set up figure
    n_markers = len(unique_markers)
    n_cols = 2  # Low and high intensity columns
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_markers, n_cols, figure=fig, hspace=0.5, wspace=0.3)
    
    for m_idx, marker in enumerate(unique_markers):
        # Get cells with this marker
        marker_cells = [cell for i, cell in enumerate(cell_data) if markers[i] == marker]
        
        if not marker_cells:
            continue
            
        # Extract tracks
        valid_tracks = []
        for cell in marker_cells:
            track = cell.get('track', [])
            if len(track) > 0:
                valid_tracks.append(track)
        
        if not valid_tracks:
            continue
            
        # Process tracks
        low_intensity_tracks = []
        high_intensity_tracks = []
        
        for track in valid_tracks:
            # Create mask versions
            low_mask = track < threshold
            high_mask = track >= threshold
            
            if np.any(low_mask):
                low_track = np.copy(track)
                low_track[~low_mask] = np.nan
                low_intensity_tracks.append(low_track)
                
            if np.any(high_mask):
                high_track = np.copy(track)
                high_track[~high_mask] = np.nan
                high_intensity_tracks.append(high_track)
        
        # Time points for x-axis
        x = np.arange(len(valid_tracks[0])) / 2.0  # Assuming 2Hz sampling
        
        # Plot low intensity (left column)
        ax_low = fig.add_subplot(gs[m_idx, 0])
        if low_intensity_tracks:
            low_avg = np.nanmean(low_intensity_tracks, axis=0)
            low_std = np.nanstd(low_intensity_tracks, axis=0)
            
            ax_low.plot(x, low_avg, 'b-', linewidth=1)
            ax_low.fill_between(x, low_avg - low_std, low_avg + low_std, 
                              color='b', alpha=0.3)
            ax_low.set_title(f'{marker} - Low (<{threshold}) (n={len(low_intensity_tracks)})')
        else:
            ax_low.text(0.5, 0.5, 'No low intensity signals', 
                      ha='center', va='center', transform=ax_low.transAxes)
        
        ax_low.set_ylabel('Signal')
        ax_low.set_xlabel('Time (seconds)')
        
        # Plot high intensity (right column)
        ax_high = fig.add_subplot(gs[m_idx, 1])
        if high_intensity_tracks:
            high_avg = np.nanmean(high_intensity_tracks, axis=0)
            high_std = np.nanstd(high_intensity_tracks, axis=0)
            
            ax_high.plot(x, high_avg, 'r-', linewidth=1)
            ax_high.fill_between(x, high_avg - high_std, high_avg + high_std, 
                               color='r', alpha=0.3)
            ax_high.set_title(f'{marker} - High (≥{threshold}) (n={len(high_intensity_tracks)})')
        else:
            ax_high.text(0.5, 0.5, 'No high intensity signals', 
                       ha='center', va='center', transform=ax_high.transAxes)
        
        ax_high.set_ylabel('Signal')
        ax_high.set_xlabel('Time (seconds)')
    
    plt.suptitle(f'Signal Intensity Analysis by {Group} (Split at {threshold})')
    plt.tight_layout()
    return fig

def plot_common_vs_specific_patterns(cell_data, figsize=(15, 10), Group='Lineage', corr_threshold=0.6, title=None):
    """
    Plot the most common activity pattern and marker-specific patterns 
    after removing cells that match too closely with the common pattern.
    
    Args:
        cell_data (list): List of cell dictionaries
        figsize (tuple): Figure size
        Group (str): Which group to use ('Lineage' or 'Dependancy')
        corr_threshold (float): Correlation threshold to filter out common pattern
        title (str): Optional custom title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get marker labels
    best_matches = [cell.get('best_match_'+Group, 'none') for cell in cell_data]
    unique_matches = sorted(set(best_matches))
    
    # Extract tracks across all cells
    all_tracks = []
    for cell in cell_data:
        if 'track' in cell and len(cell['track']) > 0:
            all_tracks.append(cell['track'])
    
    # Make sure all tracks have the same length
    track_length = len(all_tracks[0]) if all_tracks else 0
    all_tracks = [track for track in all_tracks if len(track) == track_length]
    
    if not all_tracks:
        print("No valid tracks found")
        return None
    
    # Calculate the global average track (most common pattern)
    global_avg_track = np.mean(all_tracks, axis=0)
    global_std_track = np.std(all_tracks, axis=0)
    
    # Create figure
    n_plots = len(unique_matches) + 1  # +1 for the global pattern
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot the global pattern first
    ax_global = fig.add_subplot(gs[0, 0])
    time_points = np.arange(len(global_avg_track)) / 2.0  # Assuming 2Hz sampling
    
    ax_global.plot(time_points, global_avg_track, 'k-', linewidth=1.5)
    ax_global.fill_between(time_points, global_avg_track - global_std_track, 
                         global_avg_track + global_std_track, color='gray', alpha=0.3)
    
    ax_global.set_title(f'Global Pattern (n={len(all_tracks)})')
    ax_global.set_xlabel('Time (seconds)')
    ax_global.set_ylabel('Signal')
    ax_global.set_ylim(bottom=0)  # Ensure y-axis doesn't go below zero
    
    # For each marker, filter out cells similar to global pattern
    plot_idx = 1  # Start from the second plot position
    for marker in unique_matches:
        # Get cells with this marker
        marker_cells = [cell for cell, m in zip(cell_data, best_matches) if m == marker]
        marker_size = len(marker_cells)
        
        # Extract valid tracks
        marker_tracks = []
        for cell in marker_cells:
            if 'track' in cell and len(cell['track']) == track_length:
                marker_tracks.append(cell['track'])
        
        if not marker_tracks:
            continue
        
        # Calculate correlations with the global pattern
        specific_tracks = []
        correlations = []
        
        for track in marker_tracks:
            # Calculate correlation coefficient
            corr = np.corrcoef(track, global_avg_track)[0, 1]
            correlations.append(corr)
            
            # Only include tracks not too similar to global pattern
            if np.isnan(corr) or abs(corr) < corr_threshold:
                specific_tracks.append(track)
        
        # Calculate average for marker-specific pattern
        if specific_tracks:
            specific_avg = np.mean(specific_tracks, axis=0)
            specific_std = np.std(specific_tracks, axis=0)
            
            # Plot the marker-specific pattern
            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            ax.plot(time_points, specific_avg, 'b-', linewidth=1.5)
            ax.fill_between(time_points, specific_avg - specific_std, 
                           specific_avg + specific_std, color='b', alpha=0.3)
            
            # For comparison, plot the global pattern (faded)
            ax.plot(time_points, global_avg_track, 'k-', linewidth=0.5, alpha=0.3)
            
            # Calculate percentage removed
            pct_removed = (len(marker_tracks) - len(specific_tracks)) / len(marker_tracks) * 100
            
            ax.set_title(f'{marker} Specific (n={len(specific_tracks)}, {pct_removed:.1f}% removed)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Signal')
            ax.set_ylim(bottom=0)  # Ensure y-axis doesn't go below zero
            
            plot_idx += 1
    
    if title is None:
        title = f'Global and {Group}-Specific Activity Patterns'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

#####################################

def add_significance_bars(ax, x_pos, y_max, sig_pairs, height_step=0.2):
        # Calculate y range for scaling
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        step_height = y_range * height_step
        
        # Sort by p-value for consistent display
        sig_pairs.sort(key=lambda x: x[1])
        
        # Add bars for each significant pair
        for i, ((g1, g2), p_val) in enumerate(sig_pairs):
            # Get x positions
            x1 = x_pos[g1]
            x2 = x_pos[g2]
            
            # Calculate bar height (higher bars for more significant differences)
            bar_height = y_max + step_height * (i+1)
            
            # Set significance marker based on p-value
            if p_val < 0.001:
                sig_symbol = '***'
            elif p_val < 0.01:
                sig_symbol = '**'
            elif p_val < 0.05:
                sig_symbol = '*'
            else:
                sig_symbol = 'ns'
            
            # Draw the bar
            bar_x = [x1, x2]
            bar_y = [bar_height, bar_height]
            ax.plot(bar_x, bar_y, 'k-', linewidth=1)
            
            # Add ticks at each end
            tick_height = step_height * 0.05
            ax.plot([x1, x1], [bar_height, bar_height-tick_height], 'k-', linewidth=1)
            ax.plot([x2, x2], [bar_height, bar_height-tick_height], 'k-', linewidth=1)
            
            # Add significance label
            ax.text((x1+x2)/2, bar_height + step_height*0.1, sig_symbol, 
                   ha='center', va='bottom', fontsize=8)

def run_statistical_analysis(feature_data, groups, group_labels):

        # Initialize results dictionary
        results = {
            'test_name': None,
            'p_value': None,
            'effect_size': None,
            'effect_name': None,
            'significant_pairs': []
        }
        
        try:
            # Check if all groups have the same values (no variance between groups)
            all_values = np.concatenate(groups)
            if np.all(all_values == all_values[0]):
                results['test_name'] = "Skipped"
                results['p_value'] = 1.0
                results['effect_size'] = 0.0
                results['effect_name'] = "None"
                return results
            # Test normality for each group
            normality_results = []
            for g in groups:
                if len(g) > 5000:
                    # Use D'Agostino-Pearson for large samples
                    _, norm_p = stats.normaltest(g)
                else:
                    # Use Shapiro-Wilk for smaller samples
                    _, norm_p = stats.shapiro(g)
                normality_results.append(norm_p > 0.05)
            
            all_normal = all(normality_results)
            
            # Test homogeneity of variance
            if len(groups) >= 2:
                # For very large groups, sample the data
                test_groups = []
                for g in groups:
                    if len(g) > 10000:
                        test_groups.append(np.random.choice(g, size=10000, replace=False))
                    else:
                        test_groups.append(g)
                
                _, levene_p = stats.levene(*test_groups)
                homogeneous_var = levene_p > 0.05
            else:
                homogeneous_var = True
            
            # With very large samples, non-parametric tests are often better
            very_large_dataset = any(len(g) > 5000 for g in groups)
            
            # Choose appropriate test based on assumptions
            if all_normal and homogeneous_var and not very_large_dataset:
                # Use ANOVA for normal data with equal variance
                f_val, p_val = stats.f_oneway(*groups)
                results['test_name'] = "ANOVA"
                results['p_value'] = p_val
                parametric = True
                # Calculate eta-squared effect size
                grand_mean = np.mean([item for sublist in groups for item in sublist])
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                results['effect_size'] = eta_squared
                results['effect_name'] = "η²"
            else:
                # Use Kruskal-Wallis for non-normal or large data
                h_val, p_val = stats.kruskal(*groups)
                results['test_name'] = "Kruskal-Wallis"
                results['p_value'] = p_val
                parametric = False
                # Calculate epsilon-squared for effect size
                n_samples = sum(len(g) for g in groups)
                effect_size = h_val / (n_samples - 1) if n_samples > 1 else 0
                results['effect_size'] = effect_size
                results['effect_name'] = "ε²"
            
            # Only do post-hoc tests if main test is significant
            if p_val < 0.05:
                if parametric:
                    # Tukey HSD for parametric data
                    posthoc_data = []
                    posthoc_labels = []
                    for idx, (label, group) in enumerate(zip(group_labels, groups)):
                        posthoc_data.extend(group)
                        posthoc_labels.extend([f'C{label}'] * len(group))
                    # Perform Tukey HSD
                    posthoc_results = pairwise_tukeyhsd(posthoc_data, posthoc_labels, alpha=0.05)
                    # Extract significant pairs
                    for i in range(len(posthoc_results.pvalues)):
                        if posthoc_results.pvalues[i] < 0.05:
                            pair = (posthoc_results.group1[i], posthoc_results.group2[i])
                            results['significant_pairs'].append((pair, posthoc_results.pvalues[i]))
                else:
                    # Try Dunn's test for non-parametric data
                    try:                        
                        # Make sure we have a DataFrame with right columns
                        if isinstance(feature_data, pd.DataFrame) and 'community_id' in feature_data.columns:
                            # Get feature name from the DataFrame (assuming only one feature column + community_id)
                            feature_col = [col for col in feature_data.columns if col != 'community_id'][0]
                            
                            # Perform Dunn's test with Bonferroni correction
                            posthoc_matrix = posthoc_dunn(feature_data, val_col=feature_col, 
                                                        group_col='community_id', p_adjust='bonferroni')
                            
                            # Extract significant pairs
                            for i in range(posthoc_matrix.shape[0]):
                                for j in range(posthoc_matrix.shape[1]):
                                    if i < j:  # Only check upper triangle
                                        if posthoc_matrix.iloc[i, j] < 0.05:
                                            g1 = f'C{posthoc_matrix.index[i]}'
                                            g2 = f'C{posthoc_matrix.columns[j]}'
                                            results['significant_pairs'].append(((g1, g2), posthoc_matrix.iloc[i, j]))
                        else:
                            # Fallback to pairwise Mann-Whitney if DataFrame structure doesn't match
                            raise ValueError("DataFrame structure not compatible with posthoc_dunn")
                            
                    except (ImportError, ModuleNotFoundError, ValueError) as e:
                        # Fallback to pairwise Mann-Whitney U tests
                        for i, (label_i, group_i) in enumerate(zip(group_labels, groups)):
                            for j, (label_j, group_j) in enumerate(zip(group_labels, groups)):
                                if i < j:  # Compare only unique pairs
                                    # Apply Bonferroni correction
                                    num_comparisons = len(group_labels) * (len(group_labels) - 1) / 2
                                    alpha_corrected = 0.05 / num_comparisons
                                    
                                    u_stat, p_value = stats.mannwhitneyu(group_i, group_j)
                                    if p_value < alpha_corrected:
                                        pair = (f'C{label_i}', f'C{label_j}')
                                        results['significant_pairs'].append((pair, p_value))
            
            return results
            
        except Exception as e:
            print(f"Statistical analysis failed: {e}")
            return results

def get_numerical_features(cell_data):
    """Helper function to get numerical features, excluding specific keys"""
    numerical_features = []
    for key in cell_data[0].keys():
        # Check if the value is numerical
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    return numerical_features

####################################

def plot_community_feature_distributions(cell_data, figsize=(14, 10), figure_index=0, max_plots_per_figure=6):
    
    # Define spatial overlap features
    spatial_features = [key for key in cell_data[0].keys() if key.endswith('_match')]

    # Get feature names, excluding special fields
    exclude_prefixes = ['track', 'coordinates', 'cell_id', 'occurrence_count', 'sync_community']
    
    numerical_features = []
    for key in cell_data[0].keys():
        # Skip non-numerical and excluded keys
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Check if the value is numerical
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Only include signal features (exclude spatial features)
            if key not in spatial_features:
                numerical_features.append(key)
    
    if not numerical_features:
        print(f"No signal features found in cell_data")
        return None, 0
    
    # Calculate total figures needed
    total_figures = (len(numerical_features) + max_plots_per_figure - 1) // max_plots_per_figure
    
    # If requested figure index is out of range, return None
    if figure_index >= total_figures:
        return None, total_figures
    
    # Get the subset of features for this figure
    start_idx = figure_index * max_plots_per_figure
    end_idx = min(start_idx + max_plots_per_figure, len(numerical_features))
    current_features = numerical_features[start_idx:end_idx]
    
    # Get community labels - ensure proper ordering (1, 2, 3, 4, ..., -1)
    labels = [cell.get('sync_community', -1) for cell in cell_data]
    unique_labels = sorted([l for l in set(labels) if l != -1])
    if -1 in set(labels):
        unique_labels.append(-1)
    
    # Create a DataFrame for easier plotting
    df_data = []
    for cell, label in zip(cell_data, labels):
        row = {feature: cell.get(feature, 0) for feature in current_features}
        row['Community'] = f'Community {label}'
        row['community_id'] = label  # Add raw label for post-hoc analysis
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid dimensions
    n_features = len(current_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Plot distributions for each feature
    for i, feature in enumerate(current_features):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot feature distribution by community using violinplot
        violinplot = sns.violinplot(x='Community', y=feature, data=df, ax=ax, inner='quartile',
                                   palette='Set3', cut=0)
        
        # Rotate x-ticks for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Extract data for statistical testing
        groups = [df[df['Community'] == f'Community {label}'][feature].values for label in unique_labels]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
        
        # Get mapping of x positions for significance bars
        x_positions = {f'C{label}': i for i, label in enumerate(unique_labels)}
        
        # Find maximum y value for positioning bars
        max_y_value = ax.get_ylim()[1]
        
        # Only perform tests if we have at least 2 communities with data
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            # Create feature-specific DataFrame for statistical testing
            stat_df = df[['community_id', feature]].dropna()
            
            # Run statistical analysis
            stats_results = run_statistical_analysis(stat_df, groups, unique_labels)
            
            # Add test results to plot if we have them
            if stats_results['test_name'] and stats_results['p_value'] is not None:
                # Format p-value display
                p_val = stats_results['p_value']
                if p_val < 0.001:
                    p_text = "p < 0.001"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.3f}"
                else:
                    p_text = f"p = {p_val:.2f}"
                
                # Add statistical test result and effect size to plot
                ax.text(0.02, 0.98, 
                        f"{stats_results['test_name']}: {p_text}\n{stats_results['effect_name']} = {stats_results['effect_size']:.3f}", 
                        transform=ax.transAxes, va='top', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Add significance bars if we have significant pairs
                if stats_results['significant_pairs']:
                    add_significance_bars(ax, x_positions, max_y_value, stats_results['significant_pairs'])
                    
                    # Adjust y-limit to accommodate all bars
                    y_max = ax.get_ylim()[1]
                    max_bar_height = max_y_value + 0.07 * (len(stats_results['significant_pairs']) + 1) * (ax.get_ylim()[1] - ax.get_ylim()[0])
                    ax.set_ylim(top=max(y_max, max_bar_height * 1.1))
    
    page_info = "" if total_figures <= 1 else f" (Page {figure_index+1}/{total_figures})"
    plt.suptitle('Feature Distributions by Community{page_info}')
    plt.tight_layout()
    return fig, total_figures

def plot_marker_feature_distributions(cell_data, figsize=(14, 10), figure_index=0, Group='Lineage', max_plots_per_figure=6):
    """
    Plot distributions of features grouped by marker types.
    """
    # Get best match labels
    best_matches = [cell.get('best_match_'+Group, 'none') for cell in cell_data]
    unique_matches = sorted(set(best_matches))
    
    # Generate colors based on the number of unique match types
    cmap = plt.cm.get_cmap('Set3', len(unique_matches))
    match_colors = {match: cmap(i) for i, match in enumerate(unique_matches)}
    if 'none' in match_colors:
        match_colors['none'] = '#949494'
    
    # Define spatial overlap features
    spatial_features = [key for key in cell_data[0].keys() if key.endswith('_match')]

    # Get feature names, excluding special fields
    exclude_prefixes = ['track', 'binary_track', 'peak_indices', 'coordinates', 'cell_id', 'sync_community']
    
    numerical_features = []
    for key in cell_data[0].keys():
        # Skip non-numerical and excluded keys
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Check if the value is numerical
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Only include signal features (exclude spatial features)
            if key not in spatial_features:
                numerical_features.append(key)
    
    if not numerical_features:
        print("No signal features found in cell_data")
        return None, 0
    
    # Calculate total figures needed
    total_figures = (len(numerical_features) + max_plots_per_figure - 1) // max_plots_per_figure
    
    # If requested figure index is out of range, return None
    if figure_index >= total_figures:
        return None, total_figures
    
    # Calculate which features to show in this figure
    start_idx = figure_index * max_plots_per_figure
    end_idx = min(start_idx + max_plots_per_figure, len(numerical_features))
    current_features = numerical_features[start_idx:end_idx]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid dimensions
    n_features = len(current_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Prepare data for feature plots
    feature_data = []
    for cell in cell_data:
        row = {feature: cell.get(feature, 0) for feature in numerical_features}
        match_type = cell.get('best_match_'+Group, 'none')
        row['Best Match'] = match_type  # For grouping
        row['match_id'] = match_type    # For statistical analysis
        feature_data.append(row)
    feature_df = pd.DataFrame(feature_data)
    
    # Plot each feature
    for i, feature in enumerate(current_features):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot feature distribution
        sns.violinplot(x='Best Match', y=feature, data=feature_df, ax=ax, inner='quartile', palette=match_colors, cut=0)
        
        # Extract data for statistical testing
        groups = [feature_df[feature_df['Best Match'] == match][feature].values for match in unique_matches]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
        
        # Get mapping of x positions for significance bars
        x_positions = {f'C{match}': i for i, match in enumerate(unique_matches)}
        # Find maximum y value for positioning bars
        max_y_value = ax.get_ylim()[1]
        
        # Only perform tests if we have at least 2 match types with data
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            # Create feature-specific DataFrame for statistical testing
            stat_df = feature_df[['match_id', feature]].dropna()
            
            # Run statistical analysis
            stats_results = run_statistical_analysis(stat_df, groups, unique_matches)
            
            # Add test results to plot if we have them
            if stats_results['test_name'] and stats_results['p_value'] is not None:
                # Format p-value display
                p_val = stats_results['p_value']
                if p_val < 0.001:
                    p_text = "p < 0.001"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.3f}"
                else:
                    p_text = f"p = {p_val:.2f}"
                
                # Add statistical test result and effect size to plot
                ax.text(0.02, 0.98, 
                        f"{stats_results['test_name']}: {p_text}\n{stats_results['effect_name']} = {stats_results['effect_size']:.3f}", 
                        transform=ax.transAxes, va='top', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))        

        # Rotate x-labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    page_info = f" (Page {figure_index+1}/{total_figures})"
    plt.suptitle(f'Feature Analysis by Marker{page_info}')
    plt.tight_layout(rect=[0, 0, 1, 0.97]) 
    
    return fig, total_figures

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

def plot_feature_similarity_matrix(feature_sim_matrix, cell_data, figsize=(12, 12), 
                                 title='Cell-Cell Feature Similarity Communities'):
    """
    Visualize the feature similarity matrix with cells ordered by feature communities.
    
    Args:
        feature_sim_matrix (numpy.ndarray): Cell-cell feature similarity matrix
        cell_data (list): List of cell dictionaries with feature community assignments
        figsize (tuple): Figure size
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract community assignments
    community_values = [cell.get('feature_community', -1) for cell in cell_data]
    unique_communities = sorted(set(community_values))
        
    # Create an order that groups cells by community
    community_order = []
    for comm in unique_communities:
        if comm != -1:  # Skip unclassified
            # Add all cells from this community
            community_order.extend([i for i, c in enumerate(community_values) if c == comm])
    
    # Add unclassified at the end if any
    community_order.extend([i for i, c in enumerate(community_values) if c == -1])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(100, 100, figure=fig)
    ax_matrix = fig.add_subplot(gs[:, :])
    
    # Reorder matrix
    ordered_matrix = feature_sim_matrix[community_order, :][:, community_order]
    
    # Plot matrix with colorbar
    sns.heatmap(ordered_matrix, ax=ax_matrix, cmap='viridis', vmin=0, vmax=1, 
              square=True, xticklabels=False, yticklabels=False,
              cbar_kws={'shrink': 0.8, 'aspect': 10})
    
    # Store boundary positions for ticks
    community_boundaries = []
    community_info = []
    
    # Draw community indicators
    ordered_communities = [community_values[i] for i in community_order]
    current_comm = ordered_communities[0]
    start_idx = 0
    
    for i, comm in enumerate(ordered_communities):
        if comm != current_comm or i == len(ordered_communities) - 1:
            # We've reached a new community or the end
            if i == len(ordered_communities) - 1 and comm == current_comm:
                end_idx = i + 1  # Include the last element
            else:
                end_idx = i
                community_boundaries.append(i)
                
            # Store community label info
            community_size = end_idx - start_idx
            mid_point = start_idx + community_size/2
            community_info.append((mid_point, current_comm, community_size))
            
            # Update for next community
            current_comm = comm
            start_idx = i
    
    # Add community labels below the matrix
    for mid_point, comm, size in community_info:
        if size > len(ordered_communities) / 20:  # Skip very small communities
            ax_matrix.text(mid_point, -0.05, f"C{comm} (n={size})", 
                        ha='center', va='bottom', fontsize=10, fontweight='normal',
                        color='black', transform=ax_matrix.get_xaxis_transform())
    
    # Add tick marks at community boundaries
    ax_matrix.set_xticks(community_boundaries)
    ax_matrix.set_xticklabels([])  # No labels for ticks
    ax_matrix.tick_params(axis='x', length=6, width=1, which='major', bottom=True, top=False)
    
    plt.title(title)
    plt.tight_layout()
    
    return fig

def plot_sync_vs_feature_communities(cell_data, figsize=(10, 8)):
    """
    Create a heatmap showing the overlap between synchronicity and feature-based communities.
    
    Args:
        cell_data (list): List of cell dictionaries with both community types
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract community assignments
    sync_communities = [cell.get('sync_community', -1) for cell in cell_data]
    feature_communities = [cell.get('feature_community', -1) for cell in cell_data]
    
    # Get unique community values
    unique_sync = sorted(set(sync_communities))
    unique_feature = sorted(set(feature_communities))
    
    # Create overlap matrix
    overlap_matrix = np.zeros((len(unique_sync), len(unique_feature)))
    
    # Calculate cell counts for each community combination
    for idx, cell in enumerate(cell_data):
        sync_comm = cell.get('sync_community', -1)
        feature_comm = cell.get('feature_community', -1)
        
        sync_idx = unique_sync.index(sync_comm)
        feature_idx = unique_feature.index(feature_comm)
        
        overlap_matrix[sync_idx, feature_idx] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create row/column labels
    sync_labels = [f"Sync C{comm}" for comm in unique_sync]
    feature_labels = [f"Feature C{comm}" for comm in unique_feature]
    
    # Plot heatmap
    sns.heatmap(overlap_matrix, ax=ax, cmap='viridis', annot=True, fmt='g',
               xticklabels=feature_labels, yticklabels=sync_labels)
    
    plt.title('Overlap Between Synchronicity and Feature Communities')
    plt.ylabel('Synchronicity Communities')
    plt.xlabel('Feature Communities')
    plt.tight_layout()
    
    return fig

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

    # Plot 2: Average Tracks 
    fig = plot_community_average_tracks(cell_data)
    output_path = os.path.join(output_folder, "02_avg_tracks_by_community.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    fig = plot_marker_average_tracks(cell_data)
    output_path = os.path.join(output_folder, "02b_avg_tracks_by_Contribution.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    fig = plot_marker_average_tracks(cell_data, Group='Dependancy')
    output_path = os.path.join(output_folder, "02c_avg_tracks_by_Dependency.tiff")
    fig.savefig(output_path, format='tiff', dpi=300)
    plt.close(fig)
    saved_files.append(output_path)

    # Add global vs. specific patterns for Lineage
    try:
        fig = plot_common_vs_specific_patterns(cell_data, Group='Lineage')
        output_path = os.path.join(output_folder, "02d_common_vs_specific_Lineage.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in common vs specific pattern analysis (Lineage): {e}")

    # Add global vs. specific patterns for Dependancy
    try:
        fig = plot_common_vs_specific_patterns(cell_data, Group='Dependancy')
        output_path = os.path.join(output_folder, "02e_common_vs_specific_Dependancy.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in common vs specific pattern analysis (Dependancy): {e}")

    # Add pattern analysis
    try:
        fig = plot_marker_activity_patterns(cell_data, Group='Lineage', n_patterns=3)
        output_path = os.path.join(output_folder, "02f_lineage_activity_patterns.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in activity pattern analysis (Lineage): {e}")
        
    try:
        fig = plot_marker_activity_patterns(cell_data, Group='Dependancy', n_patterns=3)
        output_path = os.path.join(output_folder, "02g_dependancy_activity_patterns.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in activity pattern analysis (Dependancy): {e}")

    # Add marker synchronization pattern analysis
    try:
        fig = plot_marker_sync_patterns(cell_data, sync_matrix, Group='Lineage')
        output_path = os.path.join(output_folder, "02h_lineage_sync_patterns.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in sync pattern analysis (Lineage): {e}")
    
    try:
        fig = plot_marker_sync_patterns(cell_data, sync_matrix, Group='Dependancy')
        output_path = os.path.join(output_folder, "02i_dependancy_sync_patterns.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    except Exception as e:
        print(f"Error in sync pattern analysis (Dependancy): {e}")
    
    # Add intensity-split analysis
    for threshold in [300]:
        try:
            fig = plot_intensity_split_by_marker(cell_data, threshold=threshold, Group='Lineage')
            output_path = os.path.join(output_folder, f"02j_lineage_intensity_split_{threshold}.tiff")
            fig.savefig(output_path, format='tiff', dpi=300)
            plt.close(fig)
            saved_files.append(output_path)
        except Exception as e:
            print(f"Error in intensity split analysis (Lineage, {threshold}): {e}")
            
        try:
            fig = plot_intensity_split_by_marker(cell_data, threshold=threshold, Group='Dependancy')
            output_path = os.path.join(output_folder, f"02k_dependancy_intensity_split_{threshold}.tiff")
            fig.savefig(output_path, format='tiff', dpi=300)
            plt.close(fig)
            saved_files.append(output_path)
        except Exception as e:
            print(f"Error in intensity split analysis (Dependancy, {threshold}): {e}")

    # Plot 3+: Signal Feature by Community
    fig_idx = 0
    while True:
        fig, total_figs = plot_community_feature_distributions(
            cell_data, figure_index=fig_idx, max_plots_per_figure=6)
        
        if fig is None:
            break
            
        output_path = os.path.join(output_folder, f"03_signal_features_{fig_idx+1}.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
        
        fig_idx += 1
        if fig_idx >= total_figs:
            break
    
    # Plot 4+: Signal Feature by marker
    fig_idx = 0

    while True:
        fig, total_figs = plot_marker_feature_distributions(cell_data, figure_index=fig_idx, max_plots_per_figure=6)
        
        if fig is None:
            break
            
        output_path = os.path.join(output_folder, f"04_Lineage_features_{fig_idx+1}.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
        
        fig_idx += 1
        if fig_idx >= total_figs:
            break
    fig_idx = 0
    while True:
        fig, total_figs = plot_marker_feature_distributions(cell_data, figure_index=fig_idx, Group='Dependancy', max_plots_per_figure=6)
        
        if fig is None:
            break
            
        output_path = os.path.join(output_folder, f"04b_Dependancy_features_{fig_idx+1}.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
        
        fig_idx += 1
        if fig_idx >= total_figs:
            break

    # Plot 5+: feature correlation matrix
    feature_sim_path = os.path.join(os.path.dirname(output_folder), 'feature_similarity_matrix.npy')
    if os.path.exists(feature_sim_path):
        feature_sim_matrix = np.load(feature_sim_path)
        
        # Plot feature similarity matrix with feature communities
        fig = plot_feature_similarity_matrix(feature_sim_matrix, cell_data,
                                          title='Cell-Cell Feature Similarity Communities')
        output_path = os.path.join(output_folder, "05_feature_similarity_communities.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
        
        # Also plot feature comparison with sync communities for comparison
        fig = plot_sync_vs_feature_communities(cell_data)
        output_path = os.path.join(output_folder, "06_sync_vs_feature_communities.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

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