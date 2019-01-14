
import warnings
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import glob
import os

from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import norm



def extract_data(path):
	X=[]

	listfile = os.listdir(path )
	

	for files in listfile[0:100]:
		pathandname=path+'/'+files
		try:
			hdulist = fits.open(pathandname)
			if np.shape(hdulist[0].data)==(128,128): #(128,128) size of the cropped images, this condition checks if the images have the good format
				nancheck=np.isnan(hdulist[0].data)
				
				nancheck.flatten()
				if np.any(nancheck):
					print 'nan found'
				else:

					data1=hdulist[0].data.astype('float32')
					nancheck=np.isnan(data1)
					
					
					
					#X.append((data1)/np.amax(data1))
					X.append(data1)

					
			hdulist.close()
		except IOError:
			print 'IOerror'



	return np.array(X)



if __name__ == '__main__':
	path ='/home/epfl/esavary/Documents/vaegalaxy/newcropped' #path of the folder containing the croped fits files
	X=extract_data(path)


	n = 5 # figure with 15x15 stacked images
	digit_size = 128
	figure = np.zeros((digit_size * n, digit_size * n))

	grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])

			digit = X[i*3+j*1].reshape(digit_size, digit_size)



			figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit


	hdu =fits.PrimaryHDU(figure)
	hdu.writeto('./mosaics/a3.fits')#path storage + name of the fits file with stacked images
	plt.figure(figsize=(128,128))
	plt.imshow(figure, cmap='Greys_r')

	plt.show()
