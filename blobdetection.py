

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
import numpy as np

import astropy.io.fits as pyfits
import os
import glob
from numpy import genfromtxt

imsize=60

def detectblob(imagepathfits,returnImage=False,thres=0.002):
    imageoneband = pyfits.open(imagepathfits)[1].data
    rescaled=imageoneband/(1.*np.amax(imageoneband))
    #print 'maxmin',np.amax(imageoneband),np.amin(imageoneband)
    try:
        blobs_doh = blob_dog(rescaled,  max_sigma=30, threshold=thres)
        if returnImage==True:
            return imageoneband,blobs_doh
        else:
            return blobs_doh
    except ValueError:
        print'Value error'
        if returnImage == True:
            return imageoneband,[0,0,0]
        else:
            return [0,0,0]


def plot_blobs(blobs,imagepath):
    image = pyfits.open(imagepath)[1].data





    fig, axes = plt.subplots(figsize=(9, 9))


    #for (blobs, color) in enumerate(sequence):

    axes.imshow(image, interpolation='nearest')

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
        axes.add_patch(c)
    axes.set_axis_off()

    plt.tight_layout()
    plt.show()

def return_blobs_inradius(radius,image,path,imcenter):
    blobslist = np.array(detectblob(path + image))
    centercoord=imcenter

    newblob=[]
    for it in np.arange(np.size(blobslist,0)):
        if np.abs(blobslist[it,0]-centercoord)<radius and np.abs(blobslist[it,1]-centercoord)<radius:
            newblob.append(blobslist[it,:])
    return newblob




def return_secondary_blob(blob1,blob2,imcenter):
    if np.abs(blob1[0]-imcenter)>np.abs(blob2[0]-imcenter) and (blob1[1]-imcenter)>np.abs(blob2[1]-imcenter):
        print'blob1'
        return blob1
    else:
        print'blob2'
        return blob2


def return_twoblobimages_param(radius,image,path,imcenter):
    blobslist=return_blobs_inradius(radius,image,path,imcenter)

    if len(blobslist)>1 and len(blobslist)<3:

        secondary_blob=return_secondary_blob(blobslist[0],blobslist[1],imcenter)
        plot_blobs(blobslist, path + image)
        return [image,secondary_blob[0],secondary_blob[1],secondary_blob[2]]


    else:
        print 'no blob or too much blobs',len(blobslist)

def returnbloblistfromcsv(csvlist,radius,path,imcenter):

    data = genfromtxt(csvlist, delimiter=',')
    listname=data[:,0]
    listblobcoor=[]
    for name in listname:

        file = glob.glob(path+'\\'+str(int(name))+'*')

        try:
            namefile= os.path.basename(file[0])
            if return_twoblobimages_param(radius,namefile,path,imcenter)is not None:
                listblobcoor.append(return_twoblobimages_param(radius,namefile,path,imcenter))
        except IndexError:
            print("No file found with this name")
    return listblobcoor

def circular_mask(pix, center, radius):
    Y, X = np.ogrid[:pix, :pix]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def return_mask_of_blobs(imagepath,npixels,dist_from_center_threshold, return_image=False):
    imagerow,blobs=detectblob(imagepath, returnImage=True,thres=0.001)
    maskprincipal=np.ones((npixels,npixels), dtype=bool)

    centerimage= int(npixels/2.)
    for blob in blobs:
        centerx=blob[0]
        centery = blob[1]
        radius= blob[2]
        if np.abs(centerx-centerimage)>dist_from_center_threshold and np.abs(centery-centerimage)>dist_from_center_threshold:
            smallmask= circular_mask(npixels,[centerx,centery],radius)
            maskprincipal = (maskprincipal==True) & (smallmask==False)
    if return_image==True:
        return imagerow[0:npixels,0:npixels],maskprincipal
    else:
        return maskprincipal


def plot_masked(imagepath,npixels,dist_from_center_threshold):
    image,mask=return_mask_of_blobs(imagepath,npixels,dist_from_center_threshold, return_image=True)
    image[mask==False]=0
    fig, axes = plt.subplots(figsize=(9, 9))
    axes.imshow(image, interpolation='nearest')
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':

    pathI='E:\\hSCsources\\requestHSC\\I\\'
    #image='20009371-553-cutout-HSC-I-9813-pdr1_udeep.fits'
    files = os.listdir(pathI)
    #list=returnbloblistfromcsv('highchisquare_for_blobdetection.csv',15,pathR,30)
    print np.array(list)
    pathmask='E:\\masks\\'

    #np.save('listepotentialtwoblob',np.array(list))
    i=0
    for image in files:
        i+=1
        print i
       # plot_masked(pathI+image, 60, 7)
        mask=return_mask_of_blobs(pathI+image,60, 7, return_image=False)

        np.save(pathmask+image[0:8]+'.npy',mask)


