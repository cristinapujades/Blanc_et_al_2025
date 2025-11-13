import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

from matplotlib.patches import FancyBboxPatch

input_cells = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1_cell_data_features.npy'
input_sync = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_ScrambledKO/E1/sync_matrix.npy'
output_folder = 'D:/00-BackUp_Matt/02-Recording/02-Calcium/30min_ScrambledKO/E1v2/'

#####################################

def extract_marker_info(point_labels):
    """
    Extract marker types and specific identifiers from labels.
    
    Args:
        point_labels: List of marker labels
        
    Returns:
        tuple: (marker_types, specific_names)
    """
    marker_types = []
    specific_names = []
    
    for label in point_labels:
        # Determine marker type (Gad/Glut/Other) - case-insensitive
        label_lower = label.lower()
        if 'gad' in label_lower:
            marker_types.append('Gad')
        elif 'glut' in label_lower:
            marker_types.append('Glut')
        else:
            marker_types.append('Other')
        
        # Extract specific identifier with a flexible approach
        if marker_types[-1] == 'Other':
            # For "Other" types, no specific identifier
            specific_names.append('unknown')
            continue
            
        # For Gad/Glut markers, extract the specific identifier
        parts = label.split('_')
        
        if len(parts) == 1:
            # No underscore, can't determine identifier
            specific_names.append('unknown')
        elif len(parts) == 2:
            # Handle case with one underscore: either "ID_gad" or "gad_ID"
            part0_lower = parts[0].lower()
            part1_lower = parts[1].lower()
            
            if 'gad' in part0_lower or 'glut' in part0_lower:
                # Pattern is "gad_ID" or "glut_ID"
                specific_names.append(parts[1])
            elif 'gad' in part1_lower or 'glut' in part1_lower:
                # Pattern is "ID_gad" or "ID_glut"
                specific_names.append(parts[0])
            else:
                # Neither part contains marker type (shouldn't happen)
                specific_names.append('unknown')
        else:
            # Multiple underscores, use a more complex approach
            # Take the first non-gad/glut part
            found = False
            for part in parts:
                part_lower = part.lower()
                if 'gad' not in part_lower and 'glut' not in part_lower:
                    specific_names.append(part)
                    found = True
                    break
            
            if not found:
                specific_names.append('unknown')
    
    return marker_types, specific_names

def prepare_feature_data(cell_data, exclude_prefixes, exclude_features=None):
    """
    Prepare feature data for dimensionality reduction.
    
    Args:
        cell_data: List of cell dictionaries
        exclude_prefixes: List of prefixes to exclude
        exclude_features: Optional list of features to exclude
        
    Returns:
        tuple: (X_scaled, numerical_features, valid_indices)
    """
    # Define spatial overlap features to exclude
    spatial_features = [key for key in cell_data[0].keys() if key.endswith('_match')]
    
    numerical_features = []
    for key in cell_data[0].keys():
        # Skip keys with excluded prefixes
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Skip spatial overlap features
        if key in spatial_features:
            continue
            
        # Skip explicitly excluded features
        if exclude_features and key in exclude_features:
            continue
        
        # Only include numerical features
        value = cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    if len(numerical_features) < 2:
        return None, None, None
    
    # Create feature matrix
    X = np.zeros((len(cell_data), len(numerical_features)))
    for i, cell in enumerate(cell_data):
        for j, feature in enumerate(numerical_features):
            X[i, j] = cell.get(feature, 0)
    
    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Track valid indices for all cells
    valid_indices = list(range(len(cell_data)))
    
    return X_scaled, numerical_features, valid_indices

def prepare_track_data(cell_data, point_labels, scale_tracks=True):
    """
    Prepare track data for dimensionality reduction.
    
    Args:
        cell_data: List of cell dictionaries
        point_labels: List of labels
        scale_tracks: Whether to normalize tracks
        
    Returns:
        tuple: (X_scaled, valid_indices, valid_labels)
    """
    # Extract activity tracks
    tracks = []
    valid_indices = []
    valid_labels = []
    
    # Get the expected track length from the first cell
    expected_length = len(cell_data[0]['track'])
    
    # Only use cells with matching track lengths
    for i, cell in enumerate(cell_data):
        if 'track' in cell and len(cell['track']) == expected_length:
            tracks.append(cell['track'])
            valid_indices.append(i)
            valid_labels.append(point_labels[i])
    
    if len(tracks) == 0:
        return None, None, None
    
    # Convert to numpy array
    X = np.array(tracks)
    
    # Scale each track (row-wise normalization)
    if scale_tracks:
        X_scaled = np.array([
            (row - np.mean(row)) / (np.std(row) + 1e-10)  # Avoid division by zero
            for row in X
        ])
    else:
        X_scaled = X
    
    return X_scaled, valid_indices, valid_labels

def create_color_schemes(marker_types, specific_names):
    """
    Create color schemes for marker types and specific names.
    
    Args:
        marker_types: List of marker types
        specific_names: List of specific names
        
    Returns:
        tuple: (type_colors, specific_name_colors)
    """
    # Define color schemes for types
    type_colors = {
        'Gad': '#1f77b4',    # Blue
        'Glut': '#ff7f0e',   # Orange
        'Other': '#7f7f7f'   # Gray
    }
    
    # Get unique specific names and assign colors
    unique_specific_names = sorted(set(specific_names))
    name_cmap = plt.colormaps['Set3'].resampled(max(8, len(unique_specific_names)))
    specific_name_colors = {name: name_cmap(i) for i, name in enumerate(unique_specific_names)}
    
    return type_colors, specific_name_colors

def plot_dual_colored_points(ax, X_reduced, marker_types, specific_names, type_colors, specific_name_colors):
    """
    Plot points with dual coloring (center and outline).
    
    Args:
        ax: Matplotlib axis
        X_reduced: Reduced data points
        marker_types: List of marker types
        specific_names: List of specific names
        type_colors: Dictionary of colors for marker types
        specific_name_colors: Dictionary of colors for specific names
        
    Returns:
        list: Legend elements
    """
    legend_elements = []
    plotted_combinations = {}
    
    for i in range(len(X_reduced)):
        face_color = type_colors[marker_types[i]]
        
        # Only add edge color if not "Other" marker type
        if marker_types[i] != "Other":
            edge_color = specific_name_colors[specific_names[i]]
            linewidth = 1
        else:
            edge_color = "none"
            linewidth = 0
        
        ax.scatter(
            X_reduced[i, 0],
            X_reduced[i, 1],
            c=[face_color],
            edgecolors=[edge_color],
            linewidths=linewidth,
            alpha=0.7,
            s=5
        )
        
        # Add to legend if this combination hasn't been seen yet
        key = (marker_types[i], specific_names[i])
        if key not in plotted_combinations:
            plotted_combinations[key] = True
            # Count occurrences of this combination
            combo_count = sum(1 for j in range(len(X_reduced)) 
                            if marker_types[j] == key[0] and specific_names[j] == key[1])
            
            # Create appropriate legend element based on marker type
            if marker_types[i] != "Other":
                legend_elements.append(
                    Line2D([0], [0], marker='o', color=edge_color, markerfacecolor=face_color, 
                          markersize=8, label=f'{key[1]}_{key[0]} ({combo_count})')
                )
            else:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='none', markerfacecolor=face_color, 
                          markersize=8, label=f'{key[0]} ({combo_count})')
                )
    
    return legend_elements

def plot_single_colored_points(ax, X_reduced, point_labels, unique_labels, label_colors, label_by):
    """
    Plot points with single coloring.
    
    Args:
        ax: Matplotlib axis
        X_reduced: Reduced data points
        point_labels: List of point labels
        unique_labels: List of unique labels
        label_colors: Dictionary of colors for labels
        label_by: How points are labeled ('community' or 'marker')
        
    Returns:
        list: Legend elements
    """
    legend_elements = []
    for label in unique_labels:
        # Find indices for this label
        indices = [i for i, l in enumerate(point_labels) if l == label]
        
        if not indices:
            continue
            
        color = label_colors.get(label, 'gray')
        
        ax.scatter(
            X_reduced[indices, 0], 
            X_reduced[indices, 1],
            c=[color],
            edgecolors=None,
            alpha=0.7,
            s=1
        )
        
        # Create legend element
        if label_by == 'community':
            legend_text = f'Community {label} ({len(indices)} cells)'
        else:
            legend_text = f'{label} ({len(indices)} cells)'
            
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                  markersize=8, label=legend_text)
        )
    
    return legend_elements

def create_dual_color_legend(legend_elements, marker_types, specific_names, type_colors, specific_name_colors):
    """
    Add sections to legend for dual colored points.
    
    Args:
        legend_elements: Existing legend elements
        marker_types: List of marker types
        specific_names: List of specific names
        type_colors: Dictionary of colors for marker types
        specific_name_colors: Dictionary of colors for specific names
        
    Returns:
        list: Updated legend elements
    """
    # Add separate sections for marker types and specific names
    legend_elements.append(Line2D([], [], linestyle='none', label=''))  # Spacer
    legend_elements.append(Line2D([], [], linestyle='none', label='Cell Types:'))
    
    for marker_type, color in type_colors.items():
        if marker_type in marker_types:  # Only add if this type exists in data
            legend_elements.append(
                Line2D([0], [0], marker='o', color='none', markerfacecolor=color, 
                      markersize=8, label=f'{marker_type}')
            )
    
    # Only add specific names section if we have non-Other marker types
    if any(t != "Other" for t in marker_types):
        legend_elements.append(Line2D([], [], linestyle='none', label=''))  # Spacer
        legend_elements.append(Line2D([], [], linestyle='none', label='Specific Names:'))
        
        for name, color in specific_name_colors.items():
            # Only include in legend if this name exists in data and is associated with a non-Other marker
            has_non_other = any(specific_names[i] == name and marker_types[i] != "Other" 
                             for i in range(len(specific_names)))
            
            if has_non_other and name != 'unknown':
                legend_elements.append(
                    Line2D([0], [0], marker='o', color=color, markerfacecolor='none', markeredgewidth=2,
                          markersize=8, label=f'{name}')
                )
    
    return legend_elements

def visualize_feature_relationships(fig, ax, method, data_source, cell_data, X_reduced, numerical_features, 
                                  pca=None, show_top_loadings=2, show_top_correlations=2):
    """
    Add feature loadings or correlations visualization.
    
    Args:
        fig: Matplotlib figure
        ax: Main axis
        method: Dimensionality reduction method ('pca' or 'umap')
        data_source: Data source ('features' or 'tracks')
        cell_data: List of cell dictionaries
        X_reduced: Reduced data
        numerical_features: List of feature names
        pca: PCA object (optional, for PCA method)
        show_top_loadings: Number of top loadings to show (for PCA)
        show_top_correlations: Number of top correlations to show (for UMAP)
        
    Returns:
        list: Additional legend elements
    """
    legend_elements = []
    
    if method == 'pca' and data_source == 'features' and pca is not None:
        # Get the PCA components (loadings)
        loadings = pca.components_
        loading_abs = np.abs(loadings)
        
        # Find top features for each principal component
        top_features_pc1 = np.argsort(loading_abs[0])[::-1][:show_top_loadings]
        top_features_pc2 = np.argsort(loading_abs[1])[::-1][:show_top_loadings]
        
        # Create a list of all top features
        unique_top_features = np.unique(np.concatenate([top_features_pc1, top_features_pc2]))
        
        # Get a color palette for features
        feature_cmap = plt.colormaps['tab10'].resampled(len(unique_top_features))
        
        # Add feature loadings to legend
        for i, idx in enumerate(unique_top_features):
            if loading_abs[0, idx] > 0.15 or loading_abs[1, idx] > 0.15:
                feature_name = numerical_features[idx]
                color = feature_cmap(i)
                
                # Add this feature to the legend
                legend_elements.append(
                    Line2D([0], [0], color=color, lw=2, marker='>', 
                           markersize=5, label=feature_name)
                )
        
        # Create inset for loadings visualization
        create_feature_inset(fig, loadings, unique_top_features, feature_cmap, 'PC')
        
    elif method == 'umap' and data_source == 'features':
        # Calculate correlations between original features and UMAP dimensions
        correlations = np.zeros((len(numerical_features), 2))
        
        for i, feature in enumerate(numerical_features):
            feature_values = np.array([cell.get(feature, 0) for cell in cell_data])
            
            # Calculate Spearman correlation (more robust than Pearson)
            correlations[i, 0], _ = stats.spearmanr(feature_values, X_reduced[:, 0])
            correlations[i, 1], _ = stats.spearmanr(feature_values, X_reduced[:, 1])
        
        # Find features with highest absolute correlation to UMAP dimensions
        corr_abs = np.abs(correlations)
        top_features_umap1 = np.argsort(corr_abs[:, 0])[::-1][:show_top_correlations]
        top_features_umap2 = np.argsort(corr_abs[:, 1])[::-1][:show_top_correlations]
        
        # Create a list of all top features
        unique_top_features = np.unique(np.concatenate([top_features_umap1, top_features_umap2]))
        
        # Get a color palette for features
        feature_cmap = plt.colormaps['tab10'].resampled(len(unique_top_features))
        
        # Add features to legend
        for i, idx in enumerate(unique_top_features):
            if corr_abs[idx, 0] > 0.3 or corr_abs[idx, 1] > 0.3:
                feature_name = numerical_features[idx]
                color = feature_cmap(i)
                
                # Add this feature to the legend
                legend_elements.append(
                    Line2D([0], [0], color=color, lw=2, marker='>', 
                           markersize=5, label=feature_name)
                )
                
        # Create inset for correlations using the same helper function
        create_feature_inset(fig, correlations, unique_top_features, feature_cmap, 'UMAP')
    
    return legend_elements

def create_feature_inset(fig, vectors, feature_indices, feature_cmap, axis_prefix):
    """
    Create inset with feature loading or correlation vectors.
    
    Args:
        fig: Matplotlib figure
        vectors: 2D array of feature vectors (loadings or correlations)
        feature_indices: Indices of features to plot
        feature_cmap: Color map for features
        axis_prefix: Prefix for axis labels ('PC' or 'UMAP')
    """
    # Create inset for visualization
    inset_width, inset_height = 0.15, 0.15
    inset_x, inset_y = 0.75, 0.11
    
    inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height], frameon=True)
    
    # Make background semi-transparent
    inset_ax.patch.set_alpha(0.7)
    inset_ax.patch.set_facecolor('white')
    
    # Add rounded border
    patch = FancyBboxPatch(
        (0, 0), 1, 1, transform=inset_ax.transAxes,
        facecolor='none', edgecolor=None, boxstyle='round,pad=0.5', alpha=0.7
    )
    inset_ax.add_patch(patch)
    
    # Set limits for the inset
    inset_ax.set_xlim(-1, 1)
    inset_ax.set_ylim(-1, 1)
    
    # Remove ticks
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    # Calculate scale based on maximum vector magnitude
    if axis_prefix == 'PC':
        # For PCA loadings, use a dynamic scale
        max_loading = max(np.max(np.abs(vectors[0])), np.max(np.abs(vectors[1])))
        scale = 0.8 / max_loading if max_loading > 0 else 0.8
    else:
        # For UMAP correlations, use a fixed scale
        scale = 0.8
    
    # Plot arrows in the inset
    vectors_abs = np.abs(vectors)
    
    for i, idx in enumerate(feature_indices):
        if (axis_prefix == 'PC' and (vectors_abs[0, idx] > 0.15 or vectors_abs[1, idx] > 0.15)) or \
           (axis_prefix == 'UMAP' and (vectors_abs[idx, 0] > 0.3 or vectors_abs[idx, 1] > 0.3)):
            color = feature_cmap(i)
            
            # Plot the arrow with color - handle the different vector shapes for PCA vs UMAP
            if axis_prefix == 'PC':
                # PCA loadings are in shape [2, n_features]
                inset_ax.arrow(0, 0, 
                            vectors[0, idx] * scale, 
                            vectors[1, idx] * scale, 
                            head_width=0.05, head_length=0.05, 
                            fc=color, ec=color, alpha=0.8)
            else:
                # UMAP correlations are in shape [n_features, 2]
                inset_ax.arrow(0, 0, 
                            vectors[idx, 0] * scale, 
                            vectors[idx, 1] * scale, 
                            head_width=0.05, head_length=0.05, 
                            fc=color, ec=color, alpha=0.8)
                    
    # Add axis labels in inset
    inset_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    inset_ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    inset_ax.text(0.95, 0.02, f'{axis_prefix}1', transform=inset_ax.transAxes, ha='right', va='bottom', fontsize=7)
    inset_ax.text(0.02, 0.95, f'{axis_prefix}2', transform=inset_ax.transAxes, ha='left', va='top', fontsize=7)

################################################################################

def plot_dimensionality_reduction(
        cell_data,
        method='pca', data_source='features', label_by='community',
        marker_group='Lineage', title=None, figsize=(10, 8),
        show_top_loadings=2, n_neighbors=100, min_dist=0.1,
        metric='euclidean', show_top_correlations=2, scale_tracks=True,
        random_state=42, exclude_features=None, dual_color_markers=False,
        pca_preprocess_components=None):

    # Get labels based on label_by parameter
    if label_by == 'community':
        point_labels = [cell.get('sync_community', -1) for cell in cell_data]
        unique_labels = sorted(set(point_labels))
        label_colors = {
            1: '#0173B2', 2: '#DE8F05', 3: '#029E73', 4: '#D55E00',
            5: '#CC78BC', 6: '#CA9161', 7: '#FBAFE4', -1: '#949494'
        }
        label_name = 'Community'
        dual_color_markers = False  # Force to False for communities
    else:  # label_by == 'marker'
        point_labels = [cell.get(f'best_match_{marker_group}', 'none') for cell in cell_data]
        unique_labels = sorted(set(point_labels))
        if not dual_color_markers:
            cmap = plt.colormaps['Set1'].resampled(len(unique_labels))
            label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
            if 'none' in label_colors:
                label_colors['none'] = '#949494'
        label_name = marker_group
    
    # Set default title if None
    if title is None:
        title = f"{method.upper()} of Cell {data_source.capitalize()} by {label_name}"
    
    # Prepare data based on source
    if data_source == 'features':
        exclude_prefixes = ['track', 'coordinates', 'cell_id', 'occurrence_count', 'sync_community']
        X_scaled, numerical_features, valid_indices = prepare_feature_data(
            cell_data, exclude_prefixes, exclude_features)
        
        if X_scaled is None:
            print(f"Not enough numerical features for {method}")
            return None
    else:  # data_source == 'tracks'
        X_scaled, valid_indices, valid_labels = prepare_track_data(
            cell_data, point_labels, scale_tracks)
        
        if X_scaled is None:
            print("No valid tracks found in cell data")
            return None
        
        # Update point_labels to only include valid tracks
        point_labels = valid_labels
    
    # Extract marker type and name for dual-color markers
    if dual_color_markers and label_by == 'marker':
        marker_types, specific_names = extract_marker_info(point_labels)
        type_colors, specific_name_colors = create_color_schemes(marker_types, specific_names)
    
    # Apply dimensionality reduction
    pca = None  # Initialize PCA object
    
    if method == 'pca':
        pca = PCA(n_components=2, random_state=random_state)
        X_reduced = pca.fit_transform(X_scaled)
    else:  # method == 'umap'
        # Apply PCA preprocessing if specified and using tracks
        if pca_preprocess_components is not None and data_source == 'tracks':
            print(f"Applying PCA preprocessing with {pca_preprocess_components} components...")
            pca_preprocessor = PCA(n_components=pca_preprocess_components, random_state=random_state)
            X_scaled = pca_preprocessor.fit_transform(X_scaled)
            print(f"PCA preprocessing complete. Shape: {X_scaled.shape}")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=min_dist, 
            n_components=2, 
            metric=metric, 
            random_state=random_state,
            n_jobs=1
        )
        X_reduced = reducer.fit_transform(X_scaled)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points with appropriate coloring
    if dual_color_markers and label_by == 'marker':
        legend_elements = plot_dual_colored_points(
            ax, X_reduced, marker_types, specific_names, type_colors, specific_name_colors)
    else:
        legend_elements = plot_single_colored_points(
            ax, X_reduced, point_labels, unique_labels, label_colors, label_by)
    
    # Add feature loadings or correlations if applicable (only for features, not preprocessed tracks)
    if not dual_color_markers and data_source == 'features':
        # For PCA, pass the pca object; for UMAP, pass None
        pca_obj = pca if method == 'pca' else None
        additional_legend = visualize_feature_relationships(
            fig, ax, method, data_source, cell_data, X_reduced, 
            numerical_features, pca_obj, show_top_loadings, show_top_correlations)
        legend_elements.extend(additional_legend)
    
    # Create legend with appropriate layout
    if dual_color_markers and label_by == 'marker':
        legend_elements = create_dual_color_legend(
            legend_elements, marker_types, specific_names, type_colors, specific_name_colors)
        
        # Two-column legend for dual coloring with smaller fontsize
        ax.legend(handles=legend_elements,
                loc='upper right', 
                frameon=True,
                framealpha=0.9,
                fancybox=True, 
                fontsize=7,
                ncol=2)
    else:
        # Standard legend
        ax.legend(handles=legend_elements,
                loc='upper right', 
                frameon=True,
                framealpha=0.9,
                fancybox=True, 
                fontsize=8)
    
    # Set axis labels
    if method == 'pca':
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    else:
        if pca_preprocess_components is not None and data_source == 'tracks':
            ax.set_xlabel('UMAP1 (from PCA-preprocessed tracks)')
            ax.set_ylabel('UMAP2 (from PCA-preprocessed tracks)')
        else:
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
    
    # Set title and grid
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Equal aspect ratio
    ax.set_aspect('equal', 'datalim')
    
    return fig

####################################################

def create_analysis_report(output_folder, cell_data, sync_matrix):

    saved_files = []
    
    # REMOVED: All PCA plots as requested
    
    # Plot 9: UMAP on features
    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='features', 
        label_by='marker', 
        marker_group='Lineage',
        title='UMAP of Cell Features by Lineage',
        dual_color_markers=True
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "09_Lineage_umap.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='features', 
        label_by='marker', 
        marker_group='Dependancy',
        title='UMAP of Cell Features by Dependancy',
        dual_color_markers=True
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "09b_Dependancy_umap.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)
    
    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='features', 
        label_by='marker', 
        marker_group='Time',
        title='UMAP of Cell Features by Time',
        dual_color_markers=False
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "09c_Time_umap.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    # Plot 10: Track UMAP by community (now with 40-component PCA preprocessing)
    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='tracks', 
        label_by='community',
        title='UMAP of Cell Activity Tracks (PCA-preprocessed)',
        pca_preprocess_components=40
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "10_track_umap.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    # Plot 10b: Track UMAP by marker (now with 40-component PCA preprocessing)
    fig = plot_dimensionality_reduction(
        cell_data, method='umap', data_source='tracks', 
        label_by='marker', marker_group='Lineage',
        title='UMAP of Cell Activity Tracks by Lineage (PCA-preprocessed)',
        dual_color_markers=True,
        pca_preprocess_components=40)
    
    if fig is not None:
        output_path = os.path.join(output_folder, "10b_track_umap_by_Lineage.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='tracks', 
        label_by='marker',
        marker_group='Dependancy',
        title='UMAP of Cell Activity Tracks by Dependancy (PCA-preprocessed)',
        dual_color_markers=True,
        pca_preprocess_components=40
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "10c_track_umap_by_Dependancy.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

    fig = plot_dimensionality_reduction(
        cell_data, 
        method='umap', 
        data_source='tracks', 
        label_by='marker',
        marker_group='Time',
        title='UMAP of Cell Activity Tracks by Time (PCA-preprocessed)',
        dual_color_markers=False,
        pca_preprocess_components=40
    )
    if fig is not None:
        output_path = os.path.join(output_folder, "10d_track_umap_by_Time.tiff")
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        saved_files.append(output_path)

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
    
    # Create images folder for TIFF files
    images_folder = os.path.join(output_folder, 'figures')
    os.makedirs(images_folder, exist_ok=True)
    
    # Create individual TIFF images instead of PDF report
    print("Generating TIFF images...")
    images_start = time.time()
    saved_images = create_analysis_report(images_folder, cell_data, sync_matrix)
    images_time = time.time() - images_start
    print(f"TIFF images generated in {images_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total processing completed in {total_time:.2f} seconds")