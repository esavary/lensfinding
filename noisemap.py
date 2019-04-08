
import numpy as np
import astropy.io.fits as pyfits

def estim_sigma_bkg_margin(image, margin=5, squared=False):
    """

    Returns a mean value of the background noise (as RMS value),

    removing a margin from the edges of the image

    """

    image_cut1 = image[0:margin, 0:margin]
    image_cut2 = image[0:margin, -margin-1:-1]
    image_cut3 = image[-margin-1:-1, -margin-1:-1]
    image_cut4 = image[-margin-1:-1, 0:margin]
    image_cut=np.mean(image_cut1+image_cut2+image_cut3+image_cut4)
    rms2 = np.mean(image_cut**2)

    mean = np.mean(image_cut)

    median = np.median(image_cut)

    # import matplotlib as mat

    # mat.use('TkAgg')

    # import matplotlib.pyplot as plt

    # plt.figure()

    # plt.hist(np.ravel(image_cut), bins =50)

    # plt.show()

    if squared:
        return rms2, mean, median

    rms = np.sqrt(rms2)

    return rms, mean, median


def estim_poisson_noise(image, exp_time=130, squared=False, rm_gaussian=None):
    """

    Computes the Poisson contribution to the noise of a given image

    (Gaussian approximation)

    """

    if rm_gaussian is not None:

        p_noise2 = np.abs(image - rm_gaussian) / exp_time

    else:

        print("Warning : Gaussian noise should have been removed from the data !")

        p_noise2 = np.abs(image) / exp_time

    if squared:
        return p_noise2

    p_noise = np.sqrt(p_noise2)

    return p_noise

def returnnoisemap(image,band):
    if band=='I':
        exp_time = 130

    else:
        exp_time = 70
    rms, mean, median = estim_sigma_bkg_margin(image)

    noisemap = estim_poisson_noise(image, exp_time=exp_time, squared=False, rm_gaussian=rms)

    return noisemap


if __name__ == '__main__':
    image = pyfits.open('E:\\hSCsources\\arch\\R\\20009980-873-cutout-HSC-R-9813-pdr1_udeep.fits')[1].data

    image_data = image[0:60, 0:60]
    rms, mean, median= estim_sigma_bkg_margin(image_data)
    print rms
    noisemap= returnnoisemap(image_data,'I')
    print np.shape(noisemap)
