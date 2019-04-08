
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
import astropy.io.fits as pyfits

from lenstronomy.Workflow.fitting_sequence import FittingSequence



from lenstronomy.Util import util

import os, shutil






movetopathG='E:\\hSCsources\\requestHSC\\G\\'
movetopathI='E:\\hSCsources\\requestHSC\\I\\'
movetopathR='E:\\hSCsources\\requestHSC\\R\\'

originalpath='E:\\hSCsources\\requestHSC18\\'



files = os.listdir(originalpath)
#files.sort()
for f in files:
    src = originalpath + f
    if 'I' in f:
        dst = movetopathI+f
        shutil.move(src,dst)


    if 'R' in f:
        dst = movetopathR+f
        shutil.move(src,dst)


    if 'G' in f:
        dst = movetopathG+f
        shutil.move(src,dst)

