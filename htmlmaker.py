
import numpy as np
import glob
import os

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from PIL import Image

images=os.listdir('./testsmulti')# folder containing the images (png,jpeg,...)

def writehtml(filename,title):
	i=0
	f=open(filename,'w')
	f.write('<html>'+'\n')
	f.write(' <head>'+'\n')
	f.write('  <style>'+'\n')
	f.write('  .container {margin-top:5px;margin-left:5px;}'+'\n')
	f.write('  .container ul { color: #E59934;}'+'\n')
	f.write('    .container ul li {margin:4px;padding:4px; width: 263px;height:263px;float: left;border: 1px solid lightgray;list-style:none;}'+'\n')
	f.write('    .container ul li:hover {border: 1px solid orange;}' + '\n')
	f.write(' </style>' + '\n')
	f.write(' </head>'+'\n')
	f.write('</html>'+'\n')
	f.write(' <body>' + '\n')
	f.write(' <h1 align="center">'+title+' </h1>' + '\n')
	f.write(' <div class="container">' + '\n')
	f.write(' <ul>' + '\n')

	i=0
	for image in images:
		#After src= write the path to the images you want to display
		#the images must be png or jpeg or any format readeable in a web page
		f.write('<li><img src= "/home/epfl/esavary/Documents/vaegalaxy/testsmulti/'+image+'" height="200" > <div>'+'test'+str(i)+'</div></li>'+'\n')
		f.write('<li><img src= "/home/epfl/esavary/Documents/vaegalaxy/decodedaugment/'+image+'" height="200" > <div>'+'decoded'+str(i)+'</div></li>'+'\n')
		i=i+1

	f.write(' </ul>' + '\n')
	f.write(' </div>' + '\n')
	f.write(' </body>' + '\n')
	f.write(' </html>' + '\n')
	f.close()

if __name__ == '__main__':
	writehtml('augmented.html','test images vs decoded images with augmentation') # argument 1 is the filename, argument 2 is the title on the top of the page
