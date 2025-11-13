from tifffile import TiffFile
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ThreadPoolExecutor

# Function to extract metadata from a TIFF file
def get_metadata(file_path):
    tif = TiffFile(file_path)
    imagej = tif.imagej_metadata
    ome = tif.ome_metadata
    data = tif.asarray()
    pixel_type = data.dtype
    dimension = data.shape
    dimension_order = tif.series[0].axes
    dimension_sizes = {dimension_order[i]: dimension[i] for i in range(len(dimension))}
    size_x = dimension_sizes.get('X', 'N/A')
    size_y = dimension_sizes.get('Y', 'N/A')
    size_z = dimension_sizes.get('Z', 'N/A')
    x = tif.pages[0].tags['XResolution'].value
    pixel_size_x = x[1] / x[0]
    y = tif.pages[0].tags['YResolution'].value
    pixel_size_y = y[1] / y[0]
    pixel_size_z = imagej.get('spacing', 1.0)
    physical_size_unit = imagej['unit']
    return pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order

# Function to calculate the mean of an array, returning 0 if the array is empty
def safe_mean(arr):
    if arr.size == 0:
        return 0 
    return np.mean(arr)

# Function to find coordinates of pixels matching a specific cell value
def find_coordinates(image, cell_value):
    return cell_value, np.where(image == cell_value)

# Function to generate coordinates for all unique cell values in the image
def generate_cell_coordinates(image):
    unique_values = np.unique(image)
    unique_values = unique_values[unique_values != 0]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda cell_value: find_coordinates(image, cell_value), unique_values))
    return {cell_value: coords for cell_value, coords in results}

# Function to extract and average pixel values at given coordinates
def extract_and_average(img, coordinates):
    extracted_pixels = img[coordinates]
    return safe_mean(extracted_pixels)

# Main function to process time series and compute average pixel values for each cell
def SingleCell_TimeSeries(recording_path, segmented_mask, output_file):
    timepoints = sorted([os.path.join(recording_path, tp) for tp in os.listdir(recording_path) if tp.endswith('.tif')])
    num_timepoints = len(timepoints)
    print(f"Number of timepoints: {num_timepoints}")
    image = TiffFile(segmented_mask).asarray()
    cell_coordinates = generate_cell_coordinates(image)
    cell_tracks = {cell_value: [] for cell_value in cell_coordinates}
    for img_path in timepoints:
        print(f"Processing: {img_path}")
        img = TiffFile(img_path).asarray()
        tasks = [(img, coordinates) for coordinates in cell_coordinates.values()]
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda task: extract_and_average(*task), tasks))
        for idx, cell_value in enumerate(cell_coordinates):
            cell_tracks[cell_value].append(results[idx])
    # Create a list of dictionaries to store cell data
    cell_data = [
        {'cell_id': cell_value, 'coordinates': cell_coordinates[cell_value], 'track': cell_tracks[cell_value]}
        for cell_value in cell_coordinates
    ]
    # Save the cell data to a file
    np.save(output_file, cell_data)
    return cell_tracks, cell_coordinates

# Function to generate and save a PDF of cell tracks
def plot_and_save_pdf(cell_tracks, output_pdf, cells_per_page=50):
    num_cells = len(cell_tracks)
    cell_ids = list(cell_tracks.keys())
    with PdfPages(output_pdf) as pdf:
        for start in range(0, num_cells, cells_per_page):
            end = min(start + cells_per_page, num_cells)
            fig, axes = plt.subplots(end - start, 1, figsize=(10, 2 * (end - start)), sharex=True)
            if end - start == 1:
                axes = [axes]
            for ax, cell_id in zip(axes, cell_ids[start:end]):
                values = cell_tracks[cell_id]
                ax.plot(values, label=f'Cell {cell_id}')
                ax.legend(loc='upper right')
                ax.set_ylabel('Average Pixel Value')
            axes[-1].set_xlabel('Timepoint')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

# User Input
recording_path = 'Y:/SV2/MB_25-03-22/30min_Ascl1bKO/E6/E2_scaled_reg_cropped/'
segmented_mask = 'Y:/SV2/MB_25-03-22/30min_Ascl1bKO/MAXT-filter-seg/MAXT_E6.tif_cp_masks.tif'

output_path = 'Y:/SV2/MB_25-03-22/30min_Ascl1bKO/TRACKS/'
output_file = os.path.join(output_path, 'E6_cell_tracks_and_coordinates.npy')

# Process the time series and generate cell tracks
tracks, coordinates = SingleCell_TimeSeries(recording_path, segmented_mask, output_file)

# Save the general cell tracks
plot_and_save_pdf(tracks, os.path.join(output_path, 'E6_general_cell_tracks.pdf'))
