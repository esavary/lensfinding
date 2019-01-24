
import numpy as np
import glob
import os

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from PIL import Image

images=os.listdir('./galaxyset/')# folder containing the images (png,jpeg,...)

def writehtml(filename,title):
	i=0
	f=open(filename,'w')
	f.write('<!DOCTYPE html>' + '\n')
	f.write('<html>'+'\n')
	f.write(' <head>'+'\n')
	f.write('  <style>'+'\n')


	f.write('  div.gallery {border: 1px solid #ccc;}'+'\n')
	f.write('div.gallery:hover {border: 1px solid #777;}'+'\n')
	f.write('div.gallery img {width: 100%;height: auto;}'+'\n')
	f.write('div.desc {padding: 15px;text-align: center;}' + '\n')
	f.write('* {box-sizing: border-box;}' + '\n')

	f.write('.responsive {padding: 0 6px;float: left; width: 24.99999%,}'+'\n')
	f.write('@media only screen and (max-width: 700px) {.responsive {width: 49.99999%;margin: 6px 0;}}'+'\n')
	f.write('@media only screen and (max-width: 700px) {.responsive {  width: 100%;}}' + '\n')
	f.write('.clearfix:after {content: "";display: table; clear: both;}' + '\n')
	f.write(' </style>' + '\n')
	f.write(' </head>' + '\n')
	f.write(' <body>' + '\n')

	f.write(' <h2>'+title+'</h2>' + '\n')




	i=0
	for image in images:
		#After src= write the path to the images you want to display
		#the images must be png or jpeg or any format readeable in a web page
		f.write(' <div class="responsive">' + '\n')
		f.write(' <div class="gallery">' + '\n')
		f.write(' <a target="_blank" href="./galaxyset/'+image+'">' + '\n')
		f.write('<img src= "./galaxyset/'+image+'" height="100 width="100"" > '+'\n')
		f.write(' </a>' + '\n')
		f.write(' </div>' + '\n')
		f.write(' </div>' + '\n')

		i=i+1


	f.write(' </body>' + '\n')
	f.write(' </html>' + '\n')
	f.close()

if __name__ == '__main__':
	writehtml('test.html','Images') # argument 1 is the filename, argument 2 is the title on the top of the page
