print(__import__('sys').version)
import numpy as np
import os
import sys
import tifffile as tif
import time
import mbirjax as mbir
import tomopy
import jax.numpy as jnp

out_path = "/tahoma/emsl61599/hasitha/out/"
sinogram = tif.imread("/tahoma/emsl61599/hasitha/sino.tif")
tomopy.minus_log(sinogram, out=sinogram, ncore=64)

COR = 11.75 # pixel
source_detector_dist = 870.86 # mm
source_iso_dist = 359.10969543457 #mm
voxel_size = 0.0824724311200843 #mm
# ALU is in pixels
angles = jnp.linspace(0, 2*np.pi, 1600, endpoint=False)
sinogram_shape = (1600, 2000, 2000)

cone_model = mbir.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist*voxel_size, source_iso_dist=source_iso_dist*voxel_size)
cone_model.set_params(det_channel_offset=COR, verbose=1)

print("######Printing Parameters############")
cone_model.print_params()

print('###################STARTING_RECON#####################')

time0 = time.time()
recon = cone_model.fdk_recon(sinogram, filter_name="ramp")

recon.block_until_ready()
elapsed = time.time() - time0
tif.imwrite(outpath+'fdk_recon.tif',recon)
print('#########################END##########################')
