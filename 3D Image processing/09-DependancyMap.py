import os
import numpy as np
from pathlib import Path
from tifffile import TiffFile, imread, imwrite
import concurrent.futures
import csv
import numpy as np

def determine_best_match(cell_data, feature_stacks, feature_name, min_score_threshold=0):
    # Get reference names from the stacks dictionary
    reference_names = list(feature_stacks.keys())
    
    # Safety check for empty cell data
    if len(cell_data) == 0:
        print("Warning: No cells to match")
        return cell_data
    
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

def find_tiff_files(directory):
    """
    Find all TIFF files in the specified directory (non-recursively).
    """
    tiff_files = []
    for ext in ['.tif', '.tiff']:
        tiff_files.extend(list(Path(directory).glob(f'*{ext}')))
        
    return [str(f) for f in tiff_files]

def extract_cell_coordinates_thread(tiff_path):
    """
    Extract cell coordinates from a TIFF file using thread-based parallelism.
    """
    print(f"Loading TIFF stack: {tiff_path}")
    tiff_stack = imread(tiff_path)
    print(f"TIFF stack shape: {tiff_stack.shape}")
    
    # Get unique values (excluding 0, which is background)
    unique_values = np.unique(tiff_stack)
    unique_values = unique_values[unique_values > 0]
    print(f"Found {len(unique_values)} unique cell values")
    
    # Function to process a single cell value
    def process_cell_value(cell_value):
        # Find coordinates where the stack equals this cell value
        z_coords, y_coords, x_coords = np.where(tiff_stack == cell_value)
        
        # Skip if no coordinates found
        if len(z_coords) == 0:
            return None
            
        # Create cell dictionary
        return {
            'id': int(cell_value),
            'coordinates': (z_coords, y_coords, x_coords)
        }
    
    # Process cell values in parallel with ThreadPoolExecutor
    cells = []
    print(f"Processing {len(unique_values)} unique cell values with ThreadPoolExecutor...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
        # Submit all tasks
        future_to_value = {executor.submit(process_cell_value, value): value 
                          for value in unique_values}
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_value):
            cell_value = future_to_value[future]
            try:
                cell = future.result()
                if cell is not None:
                    cells.append(cell)
                
                # Print progress
                completed += 1
                if completed % 100 == 0 or completed == len(unique_values):
                    print(f"Processed {completed}/{len(unique_values)} cell values ({completed/len(unique_values)*100:.1f}%)")
            except Exception as e:
                print(f"Error processing cell value {cell_value}: {e}")
    
    print(f"Extracted coordinates for {len(cells)} cells")
    return cells

def evaluate_cells_thread_parallel(cell_data, reference_stack_path, feature_name='reference_score'):
    """
    Evaluates cells against a reference TIFF stack using thread-based parallelism.
    This approach avoids pickling issues by using threads instead of processes.
    """
    # Load the reference stack once (shared between threads)
    print(f"Loading reference stack: {reference_stack_path}")
    reference_stack = imread(reference_stack_path)
    print(f"Reference stack shape: {reference_stack.shape}")
    
    # Reference stack is in [z, y, x] order
    z_max, y_max, x_max = reference_stack.shape
    
    # Evaluate intensity range
    non_zero_values = reference_stack[reference_stack > 0]
    if len(non_zero_values) > 0:
        min_intensity = np.min(non_zero_values)
        max_intensity = np.max(non_zero_values)
        print(f"Reference intensity range: {min_intensity} to {max_intensity}")
    else:
        print("Warning: Reference stack contains no non-zero values")
        return cell_data
    
    # Create a function to process one cell
    def process_one_cell(cell):
        result_cell = cell.copy()
        
        if 'coordinates' not in cell:
            return result_cell
            
        # Extract coordinates
        coords = cell['coordinates']
        
        if isinstance(coords, tuple) and len(coords) == 3:
            z_coords, y_coords, x_coords = coords
        elif isinstance(coords, list):
            z_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            x_coords = [c[2] for c in coords]
        else:
            return result_cell
        
        # Calculate total number of voxels
        total_voxels = len(x_coords)
        if total_voxels == 0:
            return result_cell
            
        # Process coordinates
        matching_voxels = 0
        matching_intensities = []
        
        for i in range(total_voxels):
            z, y, x = z_coords[i], y_coords[i], x_coords[i]
            
            # Bounds checking
            if 0 <= z < z_max and 0 <= y < y_max and 0 <= x < x_max:
                intensity = reference_stack[z, y, x]
                if intensity > 0:
                    matching_voxels += 1
                    matching_intensities.append(intensity)
        
        # Calculate statistics
        match_percentage = (matching_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        
        if matching_voxels > 0:
            mean_match_intensity = np.mean(matching_intensities)
            norm_intensity = ((mean_match_intensity - min_intensity) / 
                            (max_intensity - min_intensity) * 100) if max_intensity > min_intensity else 0
            combined_score = (match_percentage * 0.5) + (norm_intensity * 0.5)
        else:
            combined_score = 0
        
        result_cell[f'{feature_name}_match'] = combined_score
        return result_cell
    
    # Process cells in parallel using ThreadPoolExecutor
    total_cells = len(cell_data)
    print(f"Processing {total_cells} cells using parallel threads...")
    
    # ThreadPoolExecutor avoids pickling issues because threads share memory
    result_cells = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_one_cell, cell): i 
                         for i, cell in enumerate(cell_data)}
        
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            try:
                processed_cell = future.result()
                result_cells.append(processed_cell)
                
                # Print progress
                completed += 1
                if completed % 100 == 0 or completed == total_cells:
                    print(f"Processed {completed}/{total_cells} cells ({completed/total_cells*100:.1f}%)")
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Error processing cell {idx}: {e}")
                # Add the original cell to maintain data integrity
                result_cells.append(cell_data[idx])
    
    # Ensure original order is preserved
    cell_to_idx = {id(cell): idx for idx, cell in enumerate(cell_data)}
    result_cells.sort(key=lambda x: cell_to_idx.get(id(x), 0))
    
    return result_cells

def save_cell_counts_to_tiff(cell_counts, output_dir):
    """
    Save cell counts to a visualization TIFF file.
    
    Args:
        cell_counts (list): List of dicts containing file_name, ref_name, and cell_count
        output_dir (str): Directory to save the TIFF file
    """
    # Get unique filenames and reference names for indexing
    file_names = sorted(list(set(item['file_name'] for item in cell_counts)))
    ref_names = sorted(list(set(item['ref_name'] for item in cell_counts)))
    
    # Calculate the maximum cell count for normalization
    max_count = max(item['cell_count'] for item in cell_counts)
    
    # Create an image array with dimensions:
    # rows = number of file/reference combinations
    # columns = 3 (file name, reference name, count)
    rows = len(file_names) * len(ref_names)
    
    # Create a text visualization 
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white background image
    # Width based on text length, height based on number of rows
    font_size = 12
    row_height = font_size + 4
    col_width = max(200, max(len(f) for f in file_names) * (font_size // 2))
    
    img_width = 3 * col_width
    img_height = rows * row_height + 30  # Extra for header
    
    # Create image and drawing context
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a common font
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Draw header
    draw.text((col_width//2, 10), "File Name", fill='black', font=font)
    draw.text((col_width + col_width//2, 10), "Reference Name", fill='black', font=font)
    draw.text((2*col_width + col_width//2, 10), "Cell Count", fill='black', font=font)
    
    # Draw horizontal line under header
    draw.line([(0, 30), (img_width, 30)], fill='black', width=1)
    
    # Draw the data rows
    row_idx = 0
    for file_name in file_names:
        for ref_name in ref_names:
            y = 30 + row_idx * row_height + 2
            
            # Find the count for this combination
            count = 0
            for item in cell_counts:
                if item['file_name'] == file_name and item['ref_name'] == ref_name:
                    count = item['cell_count']
                    break
            
            # Draw file name
            draw.text((10, y), file_name, fill='black', font=font)
            
            # Draw reference name
            draw.text((col_width + 10, y), ref_name, fill='black', font=font)
            
            # Draw count
            draw.text((2*col_width + 10, y), str(count), fill='black', font=font)
            
            row_idx += 1
    
    # Save as TIFF
    tiff_path = os.path.join(output_dir, "cell_counts.tif")
    img.save(tiff_path, format='TIFF')
    print(f"Cell count visualization saved to {tiff_path}")
    
    # Also save as CSV for easy data analysis
    csv_path = os.path.join(output_dir, "cell_counts.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'ref_name', 'cell_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for count_data in cell_counts:
            writer.writerow(count_data)
    print(f"Cell count data also saved to {csv_path}")

def main():
    """
    Main function to process TIFF stacks and evaluate cells against references.
    """
    # Hardcoded paths
    input_dir = 'F:/01-Analyzed/07-Shape/Glut/ScrambledKO/'
    reference_dir = 'F:/01-Analyzed/07-Shape/Glut/Decrease/'
    output_dir = 'F:/01-Analyzed/07-Shape/Glut/NecessityMap/'
    min_score = 0.50
    
    os.makedirs(output_dir, exist_ok=True)
    # List to collect cell counts data
    all_cell_counts = []

    input_tiffs = find_tiff_files(input_dir)
    print(f"Found {len(input_tiffs)} input TIFF files")
    
    reference_tiffs = find_tiff_files(reference_dir)
    print(f"Found {len(reference_tiffs)} reference TIFF files")
    
    if len(reference_tiffs) != 4:
        print(f"Warning: Expected 4 reference TIFF files, but found {len(reference_tiffs)}")
    
    # Create a dictionary of reference stacks
    reference_stacks = {}
    for ref_tiff in reference_tiffs:
        ref_name = os.path.splitext(os.path.basename(ref_tiff))[0]
        reference_stacks[ref_name] = ref_tiff
    
    # Process each input TIFF file
    for input_tiff in input_tiffs:
        print(f"\nProcessing input TIFF: {input_tiff}")
        
        # Extract cells from the input TIFF using the thread-based function
        cells = extract_cell_coordinates_thread(input_tiff)
        
        # Skip processing if no cells were found
        if len(cells) == 0:
            print(f"Warning: No cells found in {input_tiff}, skipping")
            continue
        
        # Evaluate cells against each reference using the thread-based function
        for ref_name, ref_path in reference_stacks.items():
            print(f"Evaluating against reference: {ref_name}")
            cells = evaluate_cells_thread_parallel(cells, ref_path, feature_name=ref_name)
        
        # Determine the best match for each cell
        cells = determine_best_match(cells, reference_stacks, 'reference_score', min_score_threshold=min_score)
        
        # Get metadata from this input TIFF for output
        pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, unit, dim_order = get_metadata(input_tiff)
        
        # Create stack dimensions based on this input file
        stack_shape = (size_z, size_y, size_x)
        
        # Extract input file base name for output naming
        input_base = os.path.splitext(os.path.basename(input_tiff))[0]
        
        # Count cells per reference for this file
        for ref_name in list(reference_stacks.keys()) + ['none']:
            count = sum(1 for cell in cells if cell.get('best_match_reference_score') == ref_name)
            # Add to the overall counts list
            all_cell_counts.append({
                'file_name': input_base,
                'ref_name': ref_name,
                'cell_count': count
            })

        # Generate a binary stack for each reference
        for ref_name in reference_stacks.keys():
            # Create a new binary stack (8-bit)
            binary_stack = np.zeros(stack_shape, dtype=np.uint8)
            
            # For each cell
            for cell in cells:
                # If this cell's best match is this reference
                if cell.get('best_match_reference_score') == ref_name:
                    coords = cell['coordinates']
                    
                    if isinstance(coords, tuple) and len(coords) == 3:
                        z_coords, y_coords, x_coords = coords
                        
                        # Set voxels to 255 (white)
                        for i in range(len(z_coords)):
                            z, y, x = z_coords[i], y_coords[i], x_coords[i]
                            try:
                                binary_stack[z, y, x] = 255
                            except IndexError:
                                continue
                    elif isinstance(coords, list):
                        for coord in coords:
                            try:
                                z, y, x = coord
                                binary_stack[z, y, x] = 255
                            except (ValueError, IndexError):
                                continue
            
            # Create output filename: reference_name_inputfilename.tif
            output_path = os.path.join(output_dir, f"{ref_name}_{input_base}.tif")
            
            # Save binary stack with proper metadata
            print(f"Saving binary stack for reference {ref_name} to {output_path}")
            imwrite(output_path, binary_stack, imagej=True,
                    resolution=(1./pixel_size_x, 1./pixel_size_y),
                    metadata={'spacing': pixel_size_z, 'unit': unit, 'axes': 'ZYX'})
    
    # Save all cell count data to TIFF
    save_cell_counts_to_tiff(all_cell_counts, output_dir)
    print("Processing complete!")

if __name__ == "__main__":
    main()