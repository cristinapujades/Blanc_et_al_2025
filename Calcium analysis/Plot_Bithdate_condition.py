import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

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

def plot_birthdate_distribution_by_marker(combined_cell_data, output_folder, group='Lineage'):
    """
    Creates aesthetically pleasing violin plots showing birthdate distribution by marker type.
    
    Args:
        combined_cell_data: List of cell dictionaries
        output_folder: Folder to save output figures
        group: Which marker group to use ('Lineage' or 'Dependancy')
    
    Returns:
        Path to saved figure
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if birthdate data exists
    if not any('Birthdate' in cell for cell in combined_cell_data):
        print("No birthdate data found in cell data")
        return None
    
    # Extract birthdate and marker information
    data = []
    for cell in combined_cell_data:
        birthdate = cell.get('Birthdate')
        marker = cell.get(f'best_match_{group}', 'none')
        sample = cell.get('sample_name', 'unknown')
        
        # Only include cells with valid birthdate
        if birthdate is not None and not np.isnan(birthdate):
            data.append({
                'Birthdate': birthdate,
                'Marker': marker,
                'Sample': sample
            })
    
    if not data:
        print("No valid birthdate data found")
        return None
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data)
    
    # Get unique markers, excluding 'none'
    unique_markers = sorted([m for m in df['Marker'].unique() if m != 'none'])
    
    # If there are no valid markers, exit
    if not unique_markers:
        print("No valid markers found")
        return None
    
    # Sort markers to place those containing "Gad" or "gad" first
    def marker_sort_key(marker):
        marker_lower = marker.lower()
        if "gad" in marker_lower:
            return (0, marker)  # Tuple with 0 as first element for Gad markers
        elif "glut" in marker_lower:
            return (1, marker)  # Tuple with 1 as first element for Glut markers
        return (2, marker)      # Tuple with 2 as first element for other markers
    
    unique_markers.sort(key=marker_sort_key)
    
    # Define beautiful pastel color palette based on the original scripts
    marker_types = []
    for marker in unique_markers:
        marker_lower = marker.lower()
        if "gad" in marker_lower:
            marker_types.append("Gad")
        elif "glut" in marker_lower:
            marker_types.append("Glut")
        else:
            marker_types.append("Other")
    
    # Define colors based on marker type (using pastel palette)
    type_colors = {
        'Gad': '#9ecae1',    # Pastel blue
        'Glut': '#a1d99b', 
        'Other': '#ffeda0'   
    }
    
    marker_colors = [type_colors[m_type] for m_type in marker_types]
    
    # Create figure with a clean, modern look
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Set background style
    sns.set_style("whitegrid")
    ax.set_facecolor('#f8f9fa')
    
    # Filter DataFrame to include only cells with markers in unique_markers
    df_filtered = df[df['Marker'].isin(unique_markers)]
    
    # Define custom positions with reduced spacing
    positions = np.arange(1, len(unique_markers) + 1) * 0.7  # Reduce from 1.0 to 0.8 spacing

    # Create violin plot with custom styling
    violin_parts = plt.violinplot(
        [df_filtered[df_filtered['Marker'] == marker]['Birthdate'].values for marker in unique_markers],
        showmeans=True,
        showmedians=False,
        positions=positions
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(marker_colors[i])
        pc.set_edgecolor('none')
        pc.set_alpha(0.7)
    
    # Customize mean markers
    violin_parts['cmeans'].set_color('#555555')
    violin_parts['cmeans'].set_linewidth(1.5)
    

    # Calculate statistics for each marker
    stats = []
    for marker in unique_markers:
        marker_data = df_filtered[df_filtered['Marker'] == marker]['Birthdate']
        count = len(marker_data)
        mean = marker_data.mean() if count > 0 else np.nan
        std = marker_data.std() if count > 0 else np.nan
        stats.append((count, mean, std))
    
    # Add count and statistics annotations
    for i, (count, mean, std) in enumerate(stats):
        if not np.isnan(mean):
            # Calculate y position (slightly above the maximum birthdate value for this marker)
            marker_data = df_filtered[df_filtered['Marker'] == unique_markers[i]]['Birthdate']
            y_pos = marker_data.max() + 0.5 if len(marker_data) > 0 else df_filtered['Birthdate'].max()
    
    # Set x-axis labels with marker names
    plt.xticks(positions, unique_markers, rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20) 
    
    # Set axis labels and title
    plt.ylabel('Birthdate', fontsize=24, fontweight='bold')
     
    # Add a subtle border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#dddddd')
        spine.set_linewidth(1.5)
    
    # Set y-axis limits with some padding
    y_min = df_filtered['Birthdate'].min() - 1
    y_max = df_filtered['Birthdate'].max() + 1.5  # Extra space for annotations
    plt.ylim(y_min, y_max)    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_folder, f'birthdate_by_{group}_marker.tiff')
    plt.savefig(output_path, format='tiff', dpi=300)
    plt.close()
    
    print(f"Saved birthdate distribution plot to {output_path}")
    return output_path

def main():
    """
    Main function to load data and generate birthdate distribution plots.
    """
    # Input files (reusing the paths from the first script)
    input_files = [
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E1/E1_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E2/E2_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E3/E3_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E4/E4_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E5/E5_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E6/E6_cell_data_features.npy',
        'F:/02-Recording/02-Calcium/30min_ScrambledKO/E7/E7_cell_data_features.npy',
    ]
    
    # Output folder
    output_folder = 'F:/02-Recording/02-Calcium/30min_ScrambledKO/Combined_Analysis'
    
    print("Starting birthdate analysis...")
    start_time = time.time()
    
    # Ensure output folder exists
    birthdate_output = os.path.join(output_folder, 'birthdate_analysis')
    os.makedirs(birthdate_output, exist_ok=True)
    
    # Load data from all samples
    combined_cell_data = load_multiple_samples(input_files)
    
    # Create violin plots for both Lineage and Dependancy
    plot_birthdate_distribution_by_marker(combined_cell_data, birthdate_output, group='Lineage')
    plot_birthdate_distribution_by_marker(combined_cell_data, birthdate_output, group='Dependancy')
    
    # Report execution time
    elapsed_time = time.time() - start_time
    print(f"Birthdate analysis completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()