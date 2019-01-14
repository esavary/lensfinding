from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image

def fits_to_png(pathfiles):
	X=[]
	Y=[]
	files=open('goodgal5.txt', 'r') #list of all the fits that you want to convert in png
	listfile=files.read().split()
	#listfile=os.listdir(path)
	
	i=0
	for files in listfile:
		pathandname=path+'/'+files+'.fits' #changer
		
		try:
			hdulist = fits.open(pathandname)
			if np.shape(hdulist[0].data)==(128,128):
				nancheck=np.isnan(hdulist[0].data)
				
				nancheck.flatten()
				if np.any(nancheck):
					print 'nan found'
				else:

					data1=np.array(hdulist[0].data)
					
					
					#norm= np.around((data1-np.amin(data1))/(np.amax(data1)-np.amin(data1)*1.))
					

					plt.figure(figsize=(128,128))
					plt.imshow(data1, cmap='Greys_r')
					plt.savefig(files+".png")
					plt.close()
					
					
					
					
					
					
			i=i+1
			hdulist.close()
		except IOError:
			print 'IOerror'

if __name__ == '__main__':
	#path ='/home/epfl/esavary/Documents/spiralscropped'
	path ='/home/epfl/esavary/Documents/vaegalaxy/newcropped' #path of the cropped fits files

	fits_to_png(path)




