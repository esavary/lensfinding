

# standard python imports
import numpy as np
import os
import scipy
import astropy.io.fits as pyfits
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
# lenstronomy utility functions
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from astropy.io import fits


# import NGC1300 jpg image and decompose it

# find path to data

path='E:\\hSCsources\\arch-190206-122459\\'

filename="20001670-778-cutout-HSC-I-9813-pdr1_udeep.fits"
# read data

hdulist = fits.open(path+filename)

hdulist.info()

header=ngc_data =hdulist[0].header


print header
ngc_data =hdulist[1].data
print'shape data',np.shape(ngc_data)

# subtract the median of an edge of the image
median = np.median(ngc_data[:10, :10])
ngc_data -= median

# resize the image to square size (add zeros at the edges of the non-square bits of the image)
nx, ny = np.shape(ngc_data)
n_min = min(nx, ny)
n_max = max(nx, ny)
ngc_square = np.zeros((n_max, n_max))
x_start = int((n_max - nx)/2.)
y_start = int((n_max - ny)/2.)
ngc_square[x_start:x_start+nx, y_start:y_start+ny] = ngc_data

# we slightly convolve the image with a Gaussian convolution kernel of a few pixels (optional)
sigma = 5
ngc_conv = scipy.ndimage.filters.gaussian_filter(ngc_square, sigma, mode='nearest', truncate=6)

# we now degrate the pixel resoluton by a factor.
# This reduces the data volume and increases the spead of the Shapelet decomposition
factor = 1  # lower resolution of image with a given factor
numPix_large = int(len(ngc_conv)/factor)
n_new = int((numPix_large-1)*factor)
ngc_cut = ngc_conv[0:n_new,0:n_new]
x, y = util.make_grid(numPix=numPix_large-1, deltapix=1)  # make a coordinate grid
ngc_data_resized = image_util.re_size(ngc_cut, factor)  # re-size image to lower resolution

# now we come to the Shapelet decomposition
# we turn the image in a single 1d array
image_1d = util.image2array(ngc_data_resized)  # map 2d image in 1d data array

# we define the shapelet basis set we want the image to decompose in
n_max = 150  # choice of number of shapelet basis functions, 150 is a high resolution number, but takes long
beta = 10  # shapelet scale parameter (in units of resized pixels)

# import the ShapeletSet class
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
shapeletSet = ShapeletSet()

# decompose image and return the shapelet coefficients
coeff_ngc = shapeletSet.decomposition(image_1d, x, y, n_max, beta, 1., center_x=0, center_y=0)
print(len(coeff_ngc), 'number of coefficients')  # number of coefficients

# reconstruct NGC1300 with the shapelet coefficients
image_reconstructed = shapeletSet.function(x, y, coeff_ngc, n_max, beta, center_x=0, center_y=0)
# turn 1d array back into 2d image
image_reconstructed_2d = util.array2image(image_reconstructed)  # map 1d data vector in 2d image

f, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=False, sharey=False)

ax = axes[0]
im = ax.matshow(ngc_square, origin='lower')
ax.set_title("original image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[1]
im = ax.matshow(ngc_conv, origin='lower')
ax.set_title("convolved image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[2]
im = ax.matshow(ngc_data_resized, origin='lower')
ax.set_title("resized")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[3]
im = ax.matshow(image_reconstructed_2d, origin='lower')
ax.set_title("reconstructed")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

plt.show()



# let's start with the same galaxy, but double its physical size relative to the lensed example
# we keep the same data and quantities and a bit a more blury psf
fwhm_deconv = 0.25
psf_type = 'GAUSSIAN'
kwargs_psf_deconv = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm_deconv, kernelsize=31, deltaPix=deltaPix, truncate=20, kernel=None)
kwargs_psf_deconv_high_res = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm_deconv, kernelsize=31, deltaPix=deltaPix/high_res_factor, truncate=6, kernel=None)
psf_deconv = PSF(kwargs_psf_deconv)
psf_deconv_high_res = PSF(kwargs_psf_deconv_high_res)
# define center of the source (effectively the center of the Shapelet basis)
source_x = 0.
source_y = 0.

# define the source size (effectively the Shapelet scale parameter)
beta_model = 0.12
# use the shapelet coefficients decomposed from NGC1300, but make them 10 times brighter
coeff_deconv = coeff_ngc * 5 / deltaPix**2
kwargs_shapelet = {'n_max': n_max, 'beta': beta_model, 'amp': coeff_deconv, 'center_x': source_x, 'center_y': source_y}
source_model_list = ['SHAPELETS']
kwargs_light = [kwargs_shapelet]

#from lenstronomy.LensModel.lens_model import LensModel
#lensModel = LensModel(lens_model_list)

from lenstronomy.LightModel.light_model import LightModel
lightModel = LightModel(source_model_list)



imageModel = ImageModel(data_class=data_real, psf_class=psf_deconv, kwargs_numerics=kwargs_numerics, lens_model_class=None, lens_light_model_class=lightModel)
image_no_noise = imageModel.image(kwargs_lens=None, kwargs_source=None, kwargs_lens_light=kwargs_light, kwargs_ps=None)

poisson = image_util.add_poisson(image_no_noise, exp_time=exp_time)
bkg = image_util.add_background(image_no_noise, sigma_bkd=background_rms)
image_real = image_no_noise + poisson + bkg

imageModel_high_res = ImageModel(data_class=data_high_res, psf_class=psf_deconv_high_res, kwargs_numerics={}, lens_model_class=None, lens_light_model_class=lightModel)
image_high_res_conv = imageModel_high_res.image(kwargs_lens=None, kwargs_source=None, kwargs_lens_light=kwargs_light, kwargs_ps=None)

imageModel_high_res = ImageModel(data_class=data_high_res, psf_class=psf_no, kwargs_numerics={}, lens_model_class=None, lens_light_model_class=lightModel)
image_high_res = imageModel_high_res.image(kwargs_lens=None, kwargs_source=None, kwargs_lens_light=kwargs_light, kwargs_ps=None)

f, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=False, sharey=False)

ax = axes[0]
im = ax.matshow((image_high_res), origin='lower', extent=[0, 1, 0, 1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[1]
im = ax.matshow((image_high_res_conv), origin='lower', extent=[0, 1, 0, 1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[2]
im = ax.matshow((image_no_noise), origin='lower', extent=[0, 1, 0, 1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

ax = axes[3]
im = ax.matshow((image_real), origin='lower', extent=[0, 1, 0, 1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)

plt.show()
