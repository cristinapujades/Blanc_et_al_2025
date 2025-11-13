import numpy as np
from scipy.special import j1
from scipy.signal import fftconvolve
from tifffile import TiffFile
import tifffile
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from multiprocessing import cpu_count

def get_metadata(file_path):
    tif = TiffFile(file_path)
    imagej = tif.imagej_metadata
    ome = tif.ome_metadata

    # Image Array
    data = tif.asarray()

    # Get the type
    pixel_type=data.dtype
    print("Pixel type:", pixel_type)

    # Get the array shape
    dimension= data.shape
    print(dimension)
    # Get the dimension order
    dimension_order = tif.series[0].axes
    print(dimension_order)

    # Map dimension names to their sizes
    dimension_sizes = {}
    for i in range(len(dimension)):
        dimension_sizes[dimension_order[i]] = dimension[i]
    size_x=dimension_sizes.get('X', 'N/A')
    size_y=dimension_sizes.get('Y', 'N/A')
    size_z=dimension_sizes.get('Z', 'N/A')

    print("Image dimensions:", size_x, "x", size_y, "x", size_z)

    # Get the pixel size
    x = tif.pages[0].tags['XResolution'].value
    pixel_size_x = x[1] / x[0]
    y = tif.pages[0].tags['YResolution'].value
    pixel_size_y = y[1] / y[0]
    pixel_size_z = imagej.get('spacing', 1.0)

    print("Pixel size:","X=", pixel_size_x, "x","Y=", pixel_size_y, "x","Z=", pixel_size_z)

    # Get the physical size unit
    physical_size_unit = imagej['unit']
    print("Physical size unit:", physical_size_unit)

    return pixel_size_x, pixel_size_y, pixel_size_z, size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order

def generate_psf(size_xy, size_z, na, wavelength, pixel_size_xy, pixel_size_z, n_medium=1.47, n_specimen=1.33):
    """
    Generate a simple, theoretically accurate PSF
    """
    # Ensure odd sizes
    size_xy = size_xy + (1 - size_xy % 2)
    size_z = size_z + (1 - size_z % 2)
    
    center_xy = size_xy // 2
    center_z = size_z // 2
    
    # Create coordinates
    x = np.linspace(-center_xy, center_xy, size_xy) * pixel_size_xy
    y = np.linspace(-center_xy, center_xy, size_xy) * pixel_size_xy
    z = np.linspace(-center_z, center_z, size_z) * pixel_size_z
    
    X, Y, Z = np.meshgrid(x, y, z)
    R = np.sqrt(X**2 + Y**2)
    
    k = 2 * np.pi / wavelength
    alpha = np.arcsin(na / n_medium)
    
    # PSF model based on Born & Wolf
    psf = np.zeros_like(R)
    
    for idx in np.ndindex(psf.shape):
        r = R[idx]
        z = Z[idx]
        if r == 0 and z == 0:
            psf[idx] = 1
        else:
            u = k * z * np.sin(alpha/2)**2
            v = k * r * np.sin(alpha)
            if v == 0:
                psf[idx] = np.sinc(u/np.pi)**2
            else:
                psf[idx] = (2 * j1(v)/v)**2 * np.sinc(u/np.pi)**2
    
    # Normalize
    psf /= psf.sum()
    
    return psf.astype(np.float32)

def safe_convolve3d(image, kernel, mode='same'):
    pad_z = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2
    pad_x = kernel.shape[2] // 2
    
    # smooth reflection padding
    padded = np.pad(image, 
                    ((pad_z, pad_z), 
                     (pad_y, pad_y), 
                     (pad_x, pad_x)), 
                    mode='symmetric')
    
    result = fftconvolve(padded, kernel, mode='valid')
    
    return result

def total_variation(image):
    """Calculate total variation for regularization"""
    grad_z = np.diff(image, axis=0, append=image[-1:])
    grad_y = np.diff(image, axis=1, append=image[:,-1:])
    grad_x = np.diff(image, axis=2, append=image[:,:,-1:])
    
    return np.sum(np.sqrt(grad_z**2 + grad_y**2 + grad_x**2))

def deconvolve_stack(image_stack, psf, num_iter=20, step_size=0.99, regularization=1e-7):
    """
    Unbiased Richardson-Lucy deconvolution treating all pixels equally
    """
    # Convert to float32 and normalize
    dtype_info = np.iinfo(image_stack.dtype)
    scale = dtype_info.max
    
    image = image_stack.astype(np.float32) / scale
    psf = psf.astype(np.float32)
    
    # Normalize PSF
    psf /= psf.sum()
    psf_mirror = np.flip(psf)
    
    # Initialize with original image
    estimate = image.copy()
    
    # Fixed parameters for all pixels
    min_change = 1e-5
    
    for i in range(num_iter):
        old_estimate = estimate.copy()
        
        # Forward projection
        conv = safe_convolve3d(estimate, psf)
        conv = np.maximum(conv, 1e-12)  # Numerical stability without bias
        
        # Compute error ratio with uniform bounds
        ratio = image / conv
        ratio = np.clip(ratio, 0.8, 1.2)  # Conservative bounds applied equally
        
        # Back projection
        correction = safe_convolve3d(ratio, psf_mirror)
        
        # Minimal regularization applied uniformly
        if regularization > 0:
            tv_grad = total_variation(estimate)
            tv_grad = np.clip(tv_grad * regularization, -0.05, 0.05)
            correction = correction - tv_grad
        
        # Update with fixed step size
        update = (correction - 1.0) * step_size
        update = np.clip(update, -0.1, 0.1)  # Limit changes equally for all pixels
        
        # Update estimate
        estimate *= (1.0 + update)
        
        # Basic physical constraints
        estimate = np.clip(estimate, 0, 1)
        
        # Calculate relative change
        rel_change = np.mean(np.abs(estimate - old_estimate)) / (np.mean(estimate) + 1e-12)
        
        print(f"Iteration {i+1}/{num_iter}, relative change: {rel_change:.6f}")
        
        if rel_change < min_change and i > 2:
            print(f"Converged at iteration {i+1}")
            break
    
    # Scale back to original range
    result = np.clip(estimate * scale, 0, scale)
    return result.astype(image_stack.dtype)

def process_single_channel(input_file, output_file, channel_wavelength, na=0.7, n_medium=1.47, n_specimen=1.33, num_iter=20, step_size=0.99, regularization=1e-7):
    print(f"\nProcessing: {os.path.basename(input_file)}")    
    # Get metadata
    (pixel_size_x, pixel_size_y, pixel_size_z, 
     size_x, size_y, size_z, pixel_type, physical_size_unit, dimension_order) = get_metadata(input_file)
    
    # Load image
    with TiffFile(input_file) as tif:
        image_stack = tif.asarray()
    
    # Calculate theoretically accurate PSF size
    resolution_xy = 0.61 * channel_wavelength / na
    resolution_z = 2 * channel_wavelength * n_medium / (na * na)
    
    # Calculate PSF size in pixels (at least 3x resolution)
    xy_size = int(np.ceil(3 * resolution_xy / pixel_size_x))
    z_size = int(np.ceil(3 * resolution_z / pixel_size_z))
    
    # Ensure sizes are odd and at least 31 pixels
    xy_size = max(31, xy_size + (1 - xy_size % 2))
    z_size = max(31, z_size + (1 - z_size % 2))
    
    print(f"PSF size: xy={xy_size}, z={z_size}")
    print(f"Resolution (theoretical): xy={resolution_xy:.3f}µm, z={resolution_z:.3f}µm")
    
    # Generate PSF
    psf = generate_psf(xy_size, z_size, na, channel_wavelength,
                      pixel_size_x, pixel_size_z,
                      n_medium=n_medium, n_specimen=n_specimen)
    
    # Perform unbiased deconvolution
    deconvolved = deconvolve_stack(image_stack, psf, num_iter, step_size, regularization)
    
    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tifffile.imwrite( output_file, deconvolved, imagej=True, 
        resolution=(1./pixel_size_x, 1./pixel_size_y),
        metadata={'spacing': pixel_size_z, 'unit': 'um', 'axes': 'ZYX'})
    
    return deconvolved, psf

##################################################

def process_single_channel_with_params(params):
    """
    Wrapper function for parallel processing
    """
    try:
        return process_single_channel(**params)
    except Exception as e:
        print(f"Error processing {params['input_file']}: {str(e)}")
        return None, None

def process_folder(inputfolder, outputfolder, channels, channel_wavelengths, na=0.7, n_medium=1.47, num_iter=15, max_workers=None):
    """
    Process all files in parallel
    """
    input_path = Path(inputfolder)
    output_path = Path(outputfolder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all files to process
    processing_params = []
    for file_path in input_path.glob('*.tif'):
        for channel in channels:
            if channel.lower() in file_path.name.lower():
                params = {
                    'input_file': str(file_path),
                    'output_file': str(output_path / f"{file_path.name}"),
                    'channel_wavelength': channel_wavelengths[channel],
                    'na': na,
                    'n_medium': n_medium,
                    'num_iter': num_iter
                }
                processing_params.append(params)
    
    # Process files in parallel
    print(f"\nProcessing {len(processing_params)} files...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_channel_with_params, processing_params),
            total=len(processing_params),
            desc="Processing files"
        ))
    
    # Show results
    successful = sum(1 for r in results if r[0] is not None)
    print(f"\nProcessed {successful}/{len(processing_params)} files successfully")

##################################################

if __name__ == "__main__":
    # Define input and output folders
    inputfolder = "F:/01-Analyzed/00-Selected/Ngn1Cre x Gad1bSwitch/48hpf/"
    outputfolder = "F:/01-Analyzed/01-Deconvolution/Ngn1Cre x Gad1bSwitch/48hpf/"
    
    # Channel settings
    Channels = ['C1', 'C2', 'C3']
    channel_wavelengths = {'C1': 0.477, 'C2': 0.510, 'C3': 0.590}
    
    # Microscope settings
    na = 0.7          # Nominal numerical aperture
    n_medium = 1.47   # Refractive index of glycerol
    
    # Process all channels with parallel processing
    process_folder(
        inputfolder,
        outputfolder,
        Channels,
        channel_wavelengths,
        na=na,
        n_medium=n_medium,
        num_iter=20,  # Maximum iterations (will stop early if converged)
        max_workers= cpu_count()-2 
    )