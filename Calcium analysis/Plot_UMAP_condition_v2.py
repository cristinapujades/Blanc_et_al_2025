import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better multiprocessing
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import multiprocessing

# Standardized exclusion prefixes used throughout the code
EXCLUDE_PREFIXES = [
    'track', 'coordinates', 'cell_id', 'occurrence_count', 
    'sync_community', 'sync_variability', 'between_comm_sync', 'feature_community', 
    'within_comm_sync', 'frequency_avg', 'sample', 'Birthdate', 'spectral_entropy'
]

def normalize_features_across_samples(combined_cell_data):
    """
    Normalize features across all samples to reduce batch effects.
    This preserves the original features and adds normalized versions.
    """
    if not combined_cell_data:
        print("Warning: No cell data provided for normalization")
        return combined_cell_data
    
    # Define excluded prefixes
    exclude_prefixes = EXCLUDE_PREFIXES
    
    # Group cells by sample
    samples = {}
    for cell in combined_cell_data:
        sample_id = cell.get('sample_id')
        if sample_id not in samples:
            samples[sample_id] = []
        samples[sample_id].append(cell)
    
    print(f"Normalizing features across {len(samples)} samples")
    
    # Get common numerical features across all samples
    numerical_features = []
    spatial_features = [key for key in combined_cell_data[0].keys() if key.endswith('_match')]
    
    for key in combined_cell_data[0].keys():
        # Skip keys with excluded prefixes
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        
        # Skip spatial overlap features
        if key in spatial_features:
            continue
            
        # Only include numerical features
        value = combined_cell_data[0].get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numerical_features.append(key)
    
    print(f"Found {len(numerical_features)} numerical features to normalize")
    
    # For each sample, normalize features and add them as new features
    for sample_id, sample_cells in samples.items():
        print(f"  Normalizing sample {sample_id} with {len(sample_cells)} cells")
        
        # Create feature matrix for this sample
        X = np.zeros((len(sample_cells), len(numerical_features)))
        for i, cell in enumerate(sample_cells):
            for j, feature in enumerate(numerical_features):
                X[i, j] = cell.get(feature, 0)
        
        # Standardize features within this sample
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Add normalized features back to cells with 'norm_' prefix
        for i, cell in enumerate(sample_cells):
            for j, feature in enumerate(numerical_features):
                cell[f"norm_{feature}"] = X_scaled[i, j]
    
    return combined_cell_data

def extract_marker_info(point_labels):
    """
    Extract marker types and specific identifiers from labels.
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

def prepare_feature_data(cell_data, exclude_features=None):
    """
    Prepare feature data for dimensionality reduction.
    """
    # Check if normalized features exist
    has_normalized = any(key.startswith('norm_') for key in cell_data[0].keys())
    
    # Define spatial overlap features to exclude
    spatial_features = [key for key in cell_data[0].keys() if key.endswith('_match')]
    
    numerical_features = []
    for key in cell_data[0].keys():
        # If we have normalized features, only use those
        if has_normalized and not key.startswith('norm_'):
            continue
            
        # Skip keys with excluded prefixes
        if any(key.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
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
    
    # If we're using normalized features, they're already scaled
    if has_normalized:
        X_scaled = X
    else:
        # Otherwise, standardize features
        X_scaled = StandardScaler().fit_transform(X)
    
    # Track valid indices for all cells
    valid_indices = list(range(len(cell_data)))
    
    return X_scaled, numerical_features, valid_indices

def create_color_schemes(marker_types, specific_names):
    """
    Create color schemes for marker types and specific names.
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
            s=5
        )
        
        # Create legend element
        if label_by == 'community':
            legend_text = f'Community {label} ({len(indices)} cells)'
        elif label_by == 'sample':
            legend_text = f'Sample {label} ({len(indices)} cells)'
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

def visualize_feature_relationships(fig, ax, X_reduced, numerical_features, cell_data, show_top_correlations=4):
    """
    Add feature correlations visualization for UMAP.
    """
    legend_elements = []
    
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
            
    # Create inset for correlations
    create_feature_inset(fig, correlations, unique_top_features, feature_cmap, 'UMAP', numerical_features)
    
    return legend_elements

def create_feature_inset(fig, vectors, feature_indices, feature_cmap, axis_prefix, feature_names):
    """
    Create inset with feature correlation vectors.
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
    
    # Fixed scale for UMAP correlations
    scale = 0.8
    
    # Plot arrows in the inset
    vectors_abs = np.abs(vectors)
    
    for i, idx in enumerate(feature_indices):
        if vectors_abs[idx, 0] > 0.3 or vectors_abs[idx, 1] > 0.3:
            color = feature_cmap(i)
            
            # Plot the arrow
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

def load_multiple_samples(input_files):
    """
    Load cell data from multiple files.
    """
    combined_data = []
    
    for i, file_path in enumerate(input_files):
        print(f"Loading data from {file_path}...")
        try:
            sample_data = np.load(file_path, allow_pickle=True)
            
            # Extract sample name from file path
            sample_name = Path(file_path).stem.split('_')[0]  # Adjust based on naming pattern
            
            # Add sample identifier to each cell
            for cell in sample_data:
                cell['sample_id'] = i
                cell['sample_name'] = sample_name
                combined_data.append(cell)
                
            print(f"  Added {len(sample_data)} cells from sample {sample_name}")
        except Exception as e:
            print(f"  Error loading {file_path}: {str(e)}")
    
    print(f"Loaded {len(combined_data)} cells from {len(input_files)} samples")
    return combined_data

def create_umap_visualization(visualization_config):
    """
    Create a single UMAP visualization using pre-computed embedding.
    This function must be at module level for multiprocessing to work.
    """
    # Extract required data from config
    X_reduced = visualization_config['X_reduced']
    numerical_features = visualization_config['numerical_features']
    combined_cell_data = visualization_config['combined_cell_data']
    output_folder = visualization_config['output_folder']
    
    # Extract configuration options
    marker_group = visualization_config.get('marker_group', None)
    dual_color = visualization_config.get('dual_color', False)
    sample_coloring = visualization_config.get('sample_coloring', False)
    continuous_gradient = visualization_config.get('continuous_gradient', False)
    colormap = visualization_config.get('colormap', 'viridis')
    title = visualization_config.get('title', 'UMAP Visualization')
    filename = visualization_config.get('filename', 'umap.tiff')
    
    print(f"Starting: {title}")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get appropriate labels
        if sample_coloring:
            point_labels = [cell.get('sample_name', 'unknown') for cell in combined_cell_data]
            unique_labels = sorted(set(point_labels))
            cmap = plt.colormaps['tab20'].resampled(len(unique_labels))
            label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
        else:
            point_labels = [cell.get(f'best_match_{marker_group}', 'none') for cell in combined_cell_data]
            unique_labels = sorted(set(point_labels))
            
            if not dual_color and not continuous_gradient:
                cmap = plt.colormaps['Set1'].resampled(len(unique_labels))
                label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}
                if 'none' in label_colors:
                    label_colors['none'] = '#949494'
        
        # Create the appropriate visualization
        if continuous_gradient and marker_group == 'Time':
            # Direct access to Birthdate field in the cell data
            print("Extracting birthdate values directly from cell data")
            gradient_values = np.array([cell.get('Birthdate', np.nan) for cell in combined_cell_data])
            
            # Print diagnostic info
            valid_count = np.sum(~np.isnan(gradient_values))
            print(f"Found {valid_count} valid birthdate values from {len(gradient_values)} cells")
            if valid_count > 0:
                print(f"Birthdate range: {np.nanmin(gradient_values)} to {np.nanmax(gradient_values)}")
                print(f"Unique birthdate values: {np.unique(gradient_values[~np.isnan(gradient_values)])}")
            
            # Filter out NaN values for gradient coloring
            valid_indices = ~np.isnan(gradient_values)
            
            # Create scatter plot with continuous coloring
            scatter = ax.scatter(
                X_reduced[valid_indices, 0], 
                X_reduced[valid_indices, 1],
                c=gradient_values[valid_indices],
                cmap=plt.cm.get_cmap(colormap),
                alpha=0.7,
                s=5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{marker_group} Value')
            
            # Add legend showing sample counts
            unique_values = np.unique(gradient_values[~np.isnan(gradient_values)])
            legend_elements = []
            for val in sorted(unique_values):
                count = np.sum(gradient_values == val)
                if count > 0:
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=scatter.cmap(scatter.norm(val)), 
                            markersize=8, label=f'{val} ({count} cells)')
                    )
            
            if legend_elements:
                ax.legend(handles=legend_elements,
                        loc='upper right', 
                        frameon=True,
                        framealpha=0.9,
                        fancybox=True, 
                        fontsize=8)
                        
        elif dual_color:
            # Extract marker types and specific identifiers for dual coloring
            marker_types, specific_names = extract_marker_info(point_labels)
            type_colors, specific_name_colors = create_color_schemes(marker_types, specific_names)
            
            # More efficient approach: group points by color combination
            print(f"Processing dual-colored visualization with {len(X_reduced)} points")
            
            # Create dictionary to group points by color combination
            color_groups = {}
            for i in range(len(X_reduced)):
                marker_type = marker_types[i]
                specific_name = specific_names[i]
                
                face_color = type_colors[marker_type]
                if marker_type != "Other":
                    edge_color = specific_name_colors[specific_name]
                    linewidth = 1
                else:
                    edge_color = "none"
                    linewidth = 0
                
                # Use color combination as key
                color_key = (face_color, edge_color, linewidth)
                if color_key not in color_groups:
                    color_groups[color_key] = []
                
                color_groups[color_key].append(i)
            
            # Plot each group with a single scatter call
            print(f"Grouped into {len(color_groups)} color combinations")
            for idx, (color_key, indices) in enumerate(color_groups.items()):
                face_color, edge_color, linewidth = color_key
                
                # Progress indicator for large datasets
                if idx % 10 == 0 and len(color_groups) > 20:
                    print(f"Plotting group {idx+1}/{len(color_groups)} with {len(indices)} points")
                
                # Extract coordinates for this group
                x_coords = X_reduced[indices, 0]
                y_coords = X_reduced[indices, 1]
                
                # Plot with single scatter call
                ax.scatter(
                    x_coords, y_coords,
                    c=[face_color] * len(indices) if face_color != "none" else "none",
                    edgecolors=[edge_color] * len(indices) if edge_color != "none" else "none",
                    linewidths=linewidth,
                    alpha=0.7,
                    s=5
                )
            
            # Create legend elements
            legend_elements = []
            for marker_type in sorted(set(marker_types)):
                for specific_name in sorted(set(specific_names)):
                    # Count occurrences of this combination
                    combo_count = sum(1 for i in range(len(marker_types)) 
                                    if marker_types[i] == marker_type and specific_names[i] == specific_name)
                    
                    if combo_count == 0:
                        continue
                    
                    face_color = type_colors[marker_type]
                    
                    # Create appropriate legend element based on marker type
                    if marker_type != "Other":
                        edge_color = specific_name_colors[specific_name]
                        legend_elements.append(
                            Line2D([0], [0], marker='o', color=edge_color, markerfacecolor=face_color, 
                                markersize=8, label=f'{specific_name}_{marker_type} ({combo_count})')
                        )
                    else:
                        legend_elements.append(
                            Line2D([0], [0], marker='o', color='none', markerfacecolor=face_color, 
                                markersize=8, label=f'{marker_type} ({combo_count})')
                        )
            
            # Create dual-color legend
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
            # Single color visualization (for samples or simple categories)
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
                    s=5
                )
                
                # Create legend element
                if sample_coloring:
                    legend_text = f'Sample {label} ({len(indices)} cells)'
                else:
                    legend_text = f'{label} ({len(indices)} cells)'
                    
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=8, label=legend_text)
                )
            
            # Add feature relationships for non-dual, non-continuous visualizations
            if not dual_color and not continuous_gradient:
                additional_legend = visualize_feature_relationships(
                    fig, ax, X_reduced, numerical_features, combined_cell_data, 4)
                legend_elements.extend(additional_legend)
            
            # Standard legend
            ax.legend(handles=legend_elements,
                    loc='upper right', 
                    frameon=True,
                    framealpha=0.9,
                    fancybox=True, 
                    fontsize=8)
        
        # Set axis labels, title, and grid
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Equal aspect ratio
        ax.set_aspect('equal', 'datalim')
        
        # Save figure
        output_path = os.path.join(output_folder, filename)
        fig.savefig(output_path, format='tiff', dpi=300)
        plt.close(fig)
        print(f"Completed: {title}")
        return output_path
    except Exception as e:
        print(f"Error creating visualization '{title}': {str(e)}")
        return None

def create_combined_umap_visualizations(combined_cell_data, output_folder, use_parallel=True, max_workers=None):
    """
    Create UMAP visualizations for combined cell data.
    Features are already normalized per sample.
    """
    saved_files = []
    print("Preparing data and calculating UMAP embedding (this may take a while)...")
    
    # Extract features for UMAP
    X_scaled, numerical_features, _ = prepare_feature_data(combined_cell_data)
    
    if X_scaled is None:
        print("Not enough numerical features for UMAP")
        return []
    
    # Generate UMAP embedding
    print("Calculating UMAP embedding...")
    reducer = umap.UMAP(
        n_neighbors=200, 
        min_dist=0.01,
        spread= 2, 
        n_components=2, 
        metric='euclidean', 
        random_state=42,
        n_jobs=1
    )
    X_reduced = reducer.fit_transform(X_scaled)
    print("UMAP embedding complete.")
    
    # Define visualization configurations
    visualizations = [
        {
            'X_reduced': X_reduced,
            'numerical_features': numerical_features, 
            'combined_cell_data': combined_cell_data,
            'output_folder': output_folder,
            'marker_group': 'Lineage',
            'dual_color': True,
            'title': 'UMAP of Combined Cell Features by Lineage',
            'filename': "combined_lineage_umap.tiff"
        },
        {
            'X_reduced': X_reduced,
            'numerical_features': numerical_features,
            'combined_cell_data': combined_cell_data,
            'output_folder': output_folder,
            'marker_group': 'Dependancy',
            'dual_color': True,
            'title': 'UMAP of Combined Cell Features by Dependency',
            'filename': "combined_dependency_umap.tiff"
        },
        {
            'X_reduced': X_reduced,
            'numerical_features': numerical_features,
            'combined_cell_data': combined_cell_data,
            'output_folder': output_folder,
            'marker_group': 'Time',
            'continuous_gradient': True,
            'colormap': 'plasma',
            'title': 'UMAP of Combined Cell Features by Birthdate',
            'filename': "combined_birthdate_umap.tiff"
        },
        {
            'X_reduced': X_reduced,
            'numerical_features': numerical_features,
            'combined_cell_data': combined_cell_data,
            'output_folder': output_folder,
            'sample_coloring': True,
            'title': 'UMAP of Combined Cell Features by Sample',
            'filename': "combined_by_sample_umap.tiff"
        }
    ]
    
    # Create all visualizations
    if use_parallel and len(visualizations) > 1:
        print(f"Creating {len(visualizations)} visualizations in parallel...")
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(visualizations))
        
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(create_umap_visualization, visualizations)
    else:
        print(f"Creating {len(visualizations)} visualizations sequentially...")
        results = [create_umap_visualization(config) for config in visualizations]
    
    # Filter out None values from results
    saved_files = [f for f in results if f is not None]
    return saved_files

def main():
    """
    Main function with hardcoded paths to run the analysis.
    """
    # Performance settings
    use_parallel = True   # Set to True to use parallel processing
    max_workers = None    # None will use all available CPU cores
    
    # Hardcoded input files - add or remove files as needed
    input_files = [
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E2/E2_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E3/E3_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E4/E4_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E5/E5_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E6/E6_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E7/E7_cell_data_features.npy',
    ]
    
    # Hardcoded output folder
    output_folder = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/Combined_Analysis'
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing {len(input_files)} input files:")
    for file in input_files:
        print(f"  - {file}")
    print(f"Output directory: {output_folder}")
    
    # Start timing
    start_time = time.time()
    
    # Load and combine data
    combined_cell_data = load_multiple_samples(input_files)
    
    # Add this line to normalize features
    combined_cell_data = normalize_features_across_samples(combined_cell_data)
    
    # Generate visualizations
    saved_files = create_combined_umap_visualizations(
        combined_cell_data, 
        output_folder,
        use_parallel=use_parallel,
        max_workers=max_workers
    )
    
    # Report time
    total_time = time.time() - start_time
    print(f"All processing completed in {total_time:.2f} seconds")
    print(f"Generated {len(saved_files)} visualizations:")
    for file_path in saved_files:
        print(f"  - {file_path}")

if __name__ == '__main__':
    main()