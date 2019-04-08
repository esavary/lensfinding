
import numpy as np
import os
import scipy
import astropy.io.fits as pyfits
import scipy.ndimage
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits


from astropy.visualization import lupton_rgb

def make_rgb(rimage,gimage,bimage):

    mr=max(rimage.flatten())

    mg=max(gimage.flatten())

    mb=max(bimage.flatten())

    minval = min(min((rimage/mr).flatten()),min((gimage/mg).flatten()),min((bimage/mb).flatten()))

    maxval = max(max((rimage/mr).flatten()),max((gimage/mg).flatten()),max((bimage/mb).flatten()))

    map = lupton_rgb.AsinhZScaleMapping(rimage, gimage, bimage)

    color_image = map.make_rgb_image(rimage, gimage, bimage)
    return color_image

def plot_rgb(image):
    plt.figure(figsize=(64, 64))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig('E:\\plotfittingsersic\\png_color_files\\' + name[0:8] + '.png')






if __name__ == '__main__':
    pathR = 'E:\\hSCsources\\requestHSC\\R\\'
    pathI = 'E:\\hSCsources\\requestHSC\\I\\'
    pathG = 'E:\\hSCsources\\requestHSC\\G\\'
    files = os.listdir(pathR)

    for name in files:
        imagebrutR = pyfits.open(pathR + name)[1].data
        image_dataR = imagebrutR[0:60, 0:60]
        nidR = int(name[-36:-34])
        if nidR >= 0:

            nidI = nidR + 1
            print 'NR', nidR
            if nidI > 99:
                print 'error number'
                continue
            if nidI < 10:
                nameI = name[0:-36] + str(0) + str(nidI) + name[-34:-22] + 'I' + name[-21:]
            else:
                nameI = name[0:-36] + str(nidI) + name[-34:-22] + 'I' + name[-21:]
            imagebrutI = pyfits.open(pathI + nameI)[1].data
            image_dataI = imagebrutI[0:60, 0:60]


            if nidR < 1:
                print 'error number'
                continue

            nidG = nidR - 1
            if nidG < 10:
                nameG = name[0:-36] + str(0) + str(nidG) + name[-34:-22] + 'G' + name[-21:]
            else:
                nameG = name[0:-36] + str(nidG) + name[-34:-22] + 'G' + name[-21:]
            imagebrutG = pyfits.open(pathG + nameG)[1].data
            image_dataG = imagebrutG[0:60, 0:60]
            print nameG
        else:
            nidI = nidR - 1
            nameI = name[0:-36] + str(nidI) + name[-34:-22] + 'I' + name[-21:]
            imagebrutI = pyfits.open(pathI + nameI)[1].data
            image_dataI = imagebrutI[0:60, 0:60]
            nidG = nidR + 1
            nameG = name[0:-36] + str(nidG) + name[-34:-22] + 'G' + name[-21:]
            imagebrutG = pyfits.open(pathG + nameG)[1].data
            image_dataG = imagebrutG[0:60, 0:60]


        rgb_image=make_rgb(image_dataI,image_dataR,image_dataG)
        plot_rgb(rgb_image)