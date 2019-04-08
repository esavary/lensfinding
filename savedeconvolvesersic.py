

import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF
import numpy as np
import os
import scipy
import astropy.io.fits as pyfits
import scipy.ndimage
import matplotlib.pyplot as plt
import noisemap
from lenstronomy.Plots.output_plots import LensModelPlot
import lenstronomy.Plots.output_plots as out_plot
import lenstronomy.Util.class_creator as class_creator
import astropy.io.fits as pyfits
from lenstronomy.ImSim.MultiBand.multiband import MultiBand

import lenstronomy.Util.param_util as param_util
from timeit import default_timer as timer
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from numpy import genfromtxt

from lenstronomy.Util import util

numPix = 50  # cutout pixel size
deltaPix = 0.17

_, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
 \
    = util.make_grid_with_coordtransform(numPix=numPix,deltapix=deltaPix,subgrid_res=1,inverse=False)  # !!



psfI = pyfits.open('psfI.fits')[0].data[0:39,0:39]
psfG = pyfits.open('psfG.fits')[0].data[0:39,0:39]
psfR = pyfits.open('psfR.fits')[0].data[0:39,0:39]


psfI=np.pad(psfI,10,mode='edge')
psfR=np.pad(psfR,10,mode='edge')
psfG=np.pad(psfG,10,mode='edge')

image=np.zeros((60,60))
kwargs_data ={'image_data': image,'background_rms': 0.005,'exp_time': 90,'ra_at_xy_0': ra_at_xy_0,'dec_at_xy_0': dec_at_xy_0,'transform_pix2angle': Mpix2coord}

kwargs_psfI = {'psf_type': 'PIXEL','kernel_point_source': psfI,}

kwargs_psfG = {'psf_type': 'PIXEL','kernel_point_source': psfG,}

kwargs_psfR = {'psf_type': 'PIXEL','kernel_point_source': psfR,}
kwargs_numerics = {'subgrid_res': 1}

lens_light_model_list = ['SERSIC_ELLIPSE']
lightModel = LightModel(lens_light_model_list)
kwargs_model = {'lens_light_model_list': lens_light_model_list}

def retrievelenslightresult(resultscsv):
    results = genfromtxt(resultscsv, delimiter=',')
    #I band




    #assigner vleur dans lens light rsults

def savedeconvolvefits(lensRESI,lensRESR,lensRESG):
    lens_light_resultI=np.load(lensRESI)

    imageModelI = class_creator.create_image_model(kwargs_data, kwargs_psfI, kwargs_numerics, **kwargs_model)
    modelI = imageModelI.image([],[],lens_light_resultI,
                                   [], unconvolved=True, source_add=False,
                                   lens_light_add=True, point_source_add=False)
    pyfits.writeto('E:\\lenslightinfos\\deconvolvesersic\\'+lensRESI[18:-4]+'.fits',modelI,overwrite=True)

    lens_light_resultR = np.load(lensRESR)
    imageModelR = class_creator.create_image_model(kwargs_data, kwargs_psfR, kwargs_numerics, **kwargs_model)
    modelR = imageModelR.image([], [], lens_light_resultR,
                               [], unconvolved=True, source_add=False,
                               lens_light_add=True, point_source_add=False)
    pyfits.writeto('E:\\lenslightinfos\\deconvolvesersic\\'+lensRESR[18:-4]+'.fits', modelR,overwrite=True)

    lens_light_resultG = np.load(lensRESG)
    imageModelG = class_creator.create_image_model(kwargs_data, kwargs_psfG, kwargs_numerics, **kwargs_model)
    modelG = imageModelG.image([], [], lens_light_resultG,
                               [], unconvolved=True, source_add=False,
                               lens_light_add=True, point_source_add=False)
    pyfits.writeto('E:\\lenslightinfos\\deconvolvesersic\\'+lensRESG[18:-4]+'.fits', modelG,overwrite=True)



savedeconvolvefits('E:\\lenslightinfos\\lenslightinfoI20001670.npy','E:\\lenslightinfos\\lenslightinfoR20001670.npy','E:\\lenslightinfos\\lenslightinfoG20001670.npy')