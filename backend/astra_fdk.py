import numpy as np
import astra
from imageio import imread, imwrite
import os
import tifffile as tiff

# Configuration parameters
distance_source_origin = 359.10969543457  # Distance from source to origin (object) (mm)
distance_origin_detector = 870.86 - 359.10969543457  # Distance from origin (object)  to detector (mm)
detector_pixel_size = 0.2  # Detector pixel size (mm)
detector_rows = 2000  # Number of detector rows (pixels)
detector_cols = 2000  # Number of detector columns (pixels)
num_of_projections = 1600  # Number of projections
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)  # Projection angles

input_dir = '/pscratch/sd/h/hasitha/emsl/projections/'  # Directory containing projection images
output_dir = '/pscratch/sd/h/hasitha/emsl/out/'  # Directory to save reconstructed slices

# Load projections 61273 Yang_MONet Core 1 A bot_0001.tif
projections = np.zeros((detector_rows, num_of_projections, detector_cols), dtype=np.float32)
for i in range(1,num_of_projections+1):
    im = imread(os.path.join(input_dir, f'61273 Yang_MONet Core 1 A bot_{i:04d}.tif')).astype(np.float32)
    im /= np.max(im)  # Normalize the image
    projections[:, i-1, :] = im

# Create ASTRA projection geometry
proj_geom = astra.create_proj_geom(
    'cone',
    1, 1,
    detector_rows, detector_cols,
    angles,
    (distance_source_origin + distance_origin_detector) / detector_pixel_size,
    0
)

# Apply the center of rotation correction
proj_geom = astra.geom_postalignment(proj_geom, factor=(11.25)) #passing the COR offset

# Create ASTRA data object for projections
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Create ASTRA volume geometry
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)

# Create ASTRA data object for reconstruction
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)

# Configure the FDK algorithm
alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id

# Create and run the algorithm
algorithm_id = astra.algorithm.create(alg_cfg)
print("#######running FDK##########")
astra.algorithm.run(algorithm_id)
print("#######finnished FDK##########")

# Retrieve the reconstructed volume
reconstruction = astra.data3d.get(reconstruction_id)

# Post-processing: Set negative values to zero and normalize
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Save the reconstructed slices
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
print("#######writing recon files##########")
tiff.imwrite(os.path.join(output_dir, f'reconstruction_full.tif'), reconstruction)

for i in range(detector_rows):
    im = reconstruction[i, :, :]
    im = np.flipud(im)  # Flip the image vertically
    imwrite(os.path.join(output_dir, f'reco{i:04d}.tif'), im)

# Cleanup
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
