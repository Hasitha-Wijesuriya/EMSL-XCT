print(__import__('sys').version)
import numpy as np
import os
import sys
import tifffile as tif
import time
import mbirjax as mbir
import tomopy
import jax.numpy as jnp

out_path = "/pscratch/sd/h/hasitha/emsl/out/"
print("##########starting##############")
sinogram = tif.imread("/pscratch/sd/h/hasitha/emsl/sino.tif")
tomopy.minus_log(sinogram, out=sinogram)
sharpness = -1.0
COR = 11.75 # pixel
source_detector_dist = 870.86 # mm
source_iso_dist = 359.10969543457 #mm
voxel_size = 0.0824724311200843 #mm
# ALU is in pixels
angles = jnp.linspace(0, 2*np.pi, 1600, endpoint=False)
sinogram_shape = (1600, 2000, 2000)

cone_model = mbir.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist/voxel_size, source_iso_dist=source_iso_dist/voxel_size)
# weights = cone_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')
weights = None
cone_model.set_params(sharpness=sharpness, det_channel_offset=COR, verbose=1)
print("######Printing Parameters############")
cone_model.print_params()

print('###################STARTING_RECON#####################')
print('Starting recon')
time0 = time.time()
recon, recon_params = cone_model.recon(sinogram, weights=weights)

recon.block_until_ready()
elapsed = time.time() - time0
tif.imwrite(out_path+'mbirjax_recon.tif',recon)
print('#########################END##########################')
