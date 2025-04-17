import numpy as np
import matplotlib.pyplot as plt
import tigre
import tifffile as tiff
import tigre.algorithms.fdk as fdk
import os


# Load projection data

input_dir = '/pscratch/sd/h/hasitha/emsl/projections/'  # Directory containing projection images
output_dir = '/pscratch/sd/h/hasitha/emsl/out_tigre_sirt/'  # Directory to save reconstructed slices

print("##############reading-files#################")
# Load projections 61273 Yang_MONet Core 1 A bot_0001.tif
projections = np.zeros((1600, 2000, 2000), dtype=np.float32)# Shape: (num_angles, height, width)
for i in range(1,num_of_projections+1):
    im = imread(os.path.join(input_dir, f'61273 Yang_MONet Core 1 A bot_{i:04d}.tif')).astype(np.float32)
    # im /= np.max(im)  # Normalize the image
    projections[i-1, :, :] = im

projections /= np.max(projections)
projections = -np.log(projections)

angles = np.linspace(0, 2 * np.pi, num=1600, endpoint=False)

print("##############defining -geometry#################")
# Define scan geometry
geo = geometries.ConeGeo()
geo.DSD = 870.86  # Source-to-detector distance (mm)
geo.DSO = 359.10969543457  # Source-to-object distance (mm)
geo.nDetector = np.array([projections.shape[2], projections.shape[1]])  # (width, height)
geo.dDetector = np.array([0.2, 0.2])  # Detector pixel size (mm)
geo.nVoxel = np.array([2000, 2000, 2000])  # Number of voxels in reconstruction
geo.dVoxel = np.array([0.0824724311200843, 0.0824724311200843, 0.0824724311200843])  # Voxel size (mm)
geo.sDetector = geo.nDetector * geo.dDetector  # Detector size (mm)
geo.sVoxel = geo.nVoxel * geo.dVoxel  # Volume size (mm)
geo.offOrigin = np.array([0, 8.25, 0])  # Object offset (mm)
geo.offDetector = np.array([0, 0])  # Detector offset (mm)


# -------------------------------
# Define the system geometry
# -------------------------------
# Update these parameters with values from your cone beam X-ray system.
geom = {}
geom['Detector'] = [2000,2000]       # [width, height] in pixels (example values)
geom['dso'] = 359.10969543457       # Distance from the X-ray source to the isocenter (mm)
geom['dsd'] = 870.86               # Distance from the X-ray source to the detector (mm)
geom['angles'] = np.linspace(0, 2 * np.pi, 1600, endpoint=False)  # Projection angles
geom['offset_x'] = 0.0              # Horizontal detector offset (mm)
geom['offset_y'] = 0.0              # Vertical detector offset (mm)
geom['DetectorElementSpacing'] = [0.2, 0.2]
# -------------------------------
# Define the reconstruction volume
# -------------------------------
# Here we define a cubic volume with dimensions 256x256x256 voxels.
# Adjust the volume size and voxel spacing as required.
vol_dim = [2000, 2000, 2000]







print("##############starting-FDK-recon#################")
# Perform FDK reconstruction
reconstruction = algs.fdk(projections, geo, angles)

print("##############saving-files#################")
# Save reconstructed volume
tiff.imwrite(os.path.join(output_dir, f'reconstruction_full.tif'), reconstruction.astype(np.float32))

for i in range(2000):
    im = reconstruction[i, :, :]
    imwrite(os.path.join(output_dir, f'reco{i:04d}.tif'), im)

print("Reconstruction complete. Output saved as 'reconstruction.tif'")