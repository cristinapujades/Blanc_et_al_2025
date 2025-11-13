import tifffile
from tifffile import TiffFile, imwrite
import numpy as np
import os
import random

# Define input and output folders
inputfolder = "F:/01-Analyzed/01-Deconvolution/Ngn1Cre x Vglut2aSwitch/ScrambledKO/"
outputfolder = "F:/05-Code/04-Neuron/01-3D Processing/Training_dataset/"
Channels = ['C1', 'C2', 'C3']  # channel identifiers

# Set the desired number of planes for the training dataset
Size = 15  # Number of planes to select
crop_size = (180, 256)  # Crop size (height, width)

def get_metadata(file_path):
    tif = TiffFile(file_path)
    imagej = tif.imagej_metadata
    ome = tif.ome_metadata
    # Image Array
    data = tif.asarray()
    # Get the type
    pixel_type=data.dtype
    # Get the array shape
    dimension= data.shape
    # Get the dimension order
    dimension_order = tif.series[0].axes
    # Map dimension names to their sizes
    dimension_sizes = {}
    for i in range(len(dimension)):
        dimension_sizes[dimension_order[i]] = dimension[i]
    size_x=dimension_sizes.get('X', 'N/A')
    size_y=dimension_sizes.get('Y', 'N/A')
    size_z=dimension_sizes.get('Z', 'N/A')
    # Get the pixel size
    x = tif.pages[0].tags['XResolution'].value
    pixel_size_x = x[1] / x[0]
    y = tif.pages[0].tags['YResolution'].value
    pixel_size_y = y[1] / y[0]
    pixel_size_z = imagej.get('spacing', 1.0)
    # Get the physical size unit
    physical_size_unit = imagej['unit']
    return pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order

def random_crop_image(image, crop_size):
    """Perform a random crop on the image."""
    height, width = image.shape
    crop_height, crop_width = crop_size

    if crop_height > height or crop_width > width:
        raise ValueError("Crop size must be smaller than the image dimensions.")

    start_y = random.randint(0, height - crop_height)
    start_x = random.randint(0, width - crop_width)

    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def process_files(inputfolder, outputfolder, Size, Channels, crop_size):
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    all_files = os.listdir(inputfolder)
    tiff_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]

    for channel in Channels:
        channel_output_folder = os.path.join(outputfolder, channel)
        if not os.path.exists(channel_output_folder):
            os.makedirs(channel_output_folder)

        channel_files = [f for f in tiff_files if channel in f]
        num_files = len(channel_files)
        print(f"\nChannel {channel} has {num_files} files.")

        if num_files == 0:
            print(f"No files found for channel {channel}. Skipping.")
            continue

        selected_files = random.sample(channel_files, min(Size, num_files))

        for filename in selected_files:
            filepath = os.path.join(inputfolder, filename)
            print(f"\nProcessing file: {filename}")

            # Get metadata and data
            pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order = get_metadata(filepath)

            # Load the image data
            tif = TiffFile(filepath)
            data = tif.asarray()
            tif.close()

            # Rearrange data to match (Z, Y, X) order if necessary
            if len(data.shape) == 3:
                axis_indices = {axis: idx for idx, axis in enumerate(dimension_order)}
                order = [axis_indices[axis] for axis in ['Z', 'Y', 'X']]
                data = np.transpose(data, axes=order)
            elif len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)

            # Step 2: Select a random orientation
            orientations = ['axial', 'sagittal', 'coronal']
            selected_orientation = random.choice(orientations)

            # Step 3: Select a random plane from the chosen orientation
            if selected_orientation == 'axial':
                z = random.randint(0, data.shape[0] - 1)
                plane = data[z, :, :]
                view_name = f"axial_Z{z+1}"
                # For axial view, axes are YX
                axes = 'YX'
                resolution = (1.0 / pixel_size_x, 1.0 / pixel_size_y)
            elif selected_orientation == 'sagittal':
                x = random.randint(0, data.shape[2] - 1)
                plane = data[:, :, x]
                view_name = f"sagittal_X{x+1}"
                # For sagittal view, axes are ZY
                axes = 'ZY'
                resolution = (1.0 / pixel_size_z, 1.0 / pixel_size_y)
            elif selected_orientation == 'coronal':
                y = random.randint(0, data.shape[1] - 1)
                plane = data[:, y, :]
                view_name = f"coronal_Y{y+1}"
                # For coronal view, axes are ZX
                axes = 'ZX'
                resolution = (1.0 / pixel_size_z, 1.0 / pixel_size_x)

            # Step 4: Perform a random crop on the selected plane
            cropped_plane = random_crop_image(plane, crop_size)

            # Step 5: Save the cropped image with metadata
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{view_name}_cropped.tif"
            output_path = os.path.join(channel_output_folder, output_filename)

            # Prepare metadata
            metadata = {
                'spacing': pixel_size_z,
                'unit': physical_size_unit,
                'axes': axes
            }

            # Save the image with correct metadata
            imwrite(
                output_path,
                cropped_plane.astype(pixel_type),
                imagej=True,
                resolution=resolution,
                metadata=metadata
            )
            print(f"Saved {selected_orientation} view, cropped image as {output_filename}")

    print("\nFinished processing all channels.")

process_files(inputfolder, outputfolder, Size, Channels, crop_size)
