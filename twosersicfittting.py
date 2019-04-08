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
#from lenstronomy.ImSim.MultiBand.multiband_multimodel import MultiBandMultiModel
import lenstronomy.Util.param_util as param_util
from timeit import default_timer as timer
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util import util
from lenstronomy.LightModel.light_model import LightModel


listfile=np.load('listepotentialtwoblob.npy')
listname= listfile[:,0]


pathR='E:\\hSCsources\\arch\\R\\'
pathI='E:\\hSCsources\\arch\\I\\'
pathG='E:\\hSCsources\\arch\\G\\'
psfI = pyfits.open('psfI.fits')[0].data[0:39,0:39]
psfG = pyfits.open('psfG.fits')[0].data[0:39,0:39]
psfR = pyfits.open('psfR.fits')[0].data[0:39,0:39]
psfI=np.pad(psfI,10,mode='edge')
psfR=np.pad(psfR,10,mode='edge')
psfG=np.pad(psfG,10,mode='edge')

def circular_mask(pix, center, radius):
    Y, X = np.ogrid[:pix, :pix]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

mask = circular_mask(60,[30,30],20)

numPix = 50  # cutout pixel size
deltaPix = 0.17
exp_timeI = 130  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
fwhmI = 0.64
exp_timeR = 70  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
fwhmR = 0.62
exp_timeG = 70  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
fwhmG = 0.74
num_images = 4

_, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
 \
    = util.make_grid_with_coordtransform(numPix=numPix,deltapix=deltaPix,subgrid_res=1,inverse=False)  # !!




it=0
for name in listname:
    start = timer()


    imagebrutR = pyfits.open(pathR+name)[1].data
    image_dataR=imagebrutR[0:60,0:60]
    background_rmsR,mean,med = noisemap.estim_sigma_bkg_margin(image_dataR) # background noise per pixel
    nidR= int(name[-36:-34])

    if nidR>=0:

        nidI=nidR+1
        print 'NR', nidR
        if nidI>99:
            print 'error number'
            continue
        if nidI <10:
            nameI =name[0:-36]+str(0)+str(nidI)+ name[-34:-22] + 'I' + name[-21:]
        else:
            nameI = name[0:-36] + str(nidI) + name[-34:-22] + 'I' + name[-21:]
        imagebrutI = pyfits.open(pathI+nameI)[1].data
        image_dataI=np.clip(imagebrutI[0:60,0:60],a_min= 0.000001,a_max=100000)
        background_rmsI,mean,med = noisemap.estim_sigma_bkg_margin(image_dataI) # background noise per pixel

        if nidR<1:
            print 'error number'
            continue

        nidG = nidR- 1
        if nidG <10:
            nameG =name[0:-36]+str(0)+str(nidG)+ name[-34:-22] + 'G' + name[-21:]
        else:
            nameG = name[0:-36] +  str(nidG) + name[-34:-22] + 'G' + name[-21:]
        imagebrutG = pyfits.open(pathG+nameG)[1].data
        image_dataG=np.clip(imagebrutG[0:60,0:60],a_min= 0.000001,a_max=100000)
        background_rmsG,mean,med = noisemap.estim_sigma_bkg_margin(image_dataG) # background noise per pixel
        print nameG
    else:
        nidI = nidR - 1
        nameI = name[0:-36] + str(nidI) + name[-34:-22] + 'I' + name[-21:]
        imagebrutI = pyfits.open(pathI + nameI)[1].data
        image_dataI = imagebrutI[0:60, 0:60]
        background_rmsI, mean, med = noisemap.estim_sigma_bkg_margin(image_dataI)
        nidG = nidR + 1
        nameG = name[0:-36] + str(nidG) + name[-34:-22] + 'G' + name[-21:]
        imagebrutG = pyfits.open(pathG + nameG)[1].data
        image_dataG = np.clip(imagebrutG[0:60, 0:60],a_min= 0.000001,a_max=100000)

        background_rmsG, mean, med = noisemap.estim_sigma_bkg_margin(image_dataG)


    kwargs_dataI = {'image_data': image_dataI,'background_rms': background_rmsI,'exp_time': exp_timeI,
        'ra_at_xy_0': ra_at_xy_0,'dec_at_xy_0': dec_at_xy_0,'transform_pix2angle': Mpix2coord}

    kwargs_dataR = {'image_data': image_dataR,'background_rms': background_rmsR,'exp_time': exp_timeI,
        'ra_at_xy_0': ra_at_xy_0,'dec_at_xy_0': dec_at_xy_0,'transform_pix2angle': Mpix2coord}

    kwargs_dataG = {'image_data': image_dataG,'background_rms': background_rmsG,'exp_time': exp_timeG,
                    'ra_at_xy_0': ra_at_xy_0,'dec_at_xy_0': dec_at_xy_0,'transform_pix2angle': Mpix2coord }

    kwargs_psfI = {'psf_type': 'PIXEL', 'kernel_point_source': psfI, }

    kwargs_psfG = {'psf_type': 'PIXEL', 'kernel_point_source': psfG, }

    kwargs_psfR = {'psf_type': 'PIXEL', 'kernel_point_source': psfR, }
    kwargs_numerics = {'subgrid_res': 1}

    lens_light_model_listI = ['SERSIC_ELLIPSE','SERSIC_ELLIPSE']
    lens_light_model_listG = ['SERSIC_ELLIPSE','SERSIC_ELLIPSE']
    lens_light_model_listR = ['SERSIC_ELLIPSE','SERSIC_ELLIPSE']
    lightModelI = LightModel(lens_light_model_listI)
    lightModelR = LightModel(lens_light_model_listR)
    lightModelG = LightModel(lens_light_model_listG)

    kwargs_numerics = {'subgrid_res': 1, 'mask': mask}


    kwargs_modelI = {'lens_light_model_list': lens_light_model_listI}
    kwargs_modelR = {'lens_light_model_list': lens_light_model_listR}
    kwargs_modelG = {'lens_light_model_list': lens_light_model_listG}
    kwargs_constraints = {}
    kwargs_numerics_galfit = {'subgrid_res': 1}
    kwargs_likelihood = {'check_bounds': True,  'check_positive_flux':True }

    kwargs_selected_modelsI = {'index_lens_light_model': [0]}
    kwargs_selected_modelsR = {'index_lens_light_model': [0]}
    kwargs_selected_modelsG = {'index_lens_light_model': [0]}


    image_bandI = [kwargs_dataI, kwargs_psfI, kwargs_numerics_galfit,kwargs_selected_modelsI]
    image_bandG = [kwargs_dataG, kwargs_psfG, kwargs_numerics_galfit,kwargs_selected_modelsG]
    image_bandR = [kwargs_dataR, kwargs_psfI, kwargs_numerics_galfit,kwargs_selected_modelsR]

    multi_band_listI = [image_bandI]
    multi_band_listR = [image_bandR]
    multi_band_listG = [image_bandG]
    # lens light model choices
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    #parameters extracted with blobdetection:
    radpix_secondblob=float(listfile[it,3])
    x0blob=float(listfile[it,1])
    y0blob = float(listfile[it, 2])

    # first Sersic component
    fixed_lens_light.append({})
    kwargs_lens_light_init.append({'R_sersic': 8*deltaPix, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
    kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.3, 'e1': 0.05, 'e2': 0.05, 'center_x': 0.05, 'center_y': 0.05})
    kwargs_lower_lens_light.append({'e1': -0.25, 'e2': -0.25, 'R_sersic': 3*deltaPix, 'n_sersic': 0.5, 'center_x': -4*deltaPix, 'center_y': -4*deltaPix})
    kwargs_upper_lens_light.append({'e1': 0.25, 'e2': 0.25, 'R_sersic': 40*deltaPix, 'n_sersic': 6, 'center_x': 4*deltaPix, 'center_y': 4*deltaPix})

    # second Sersic component
    fixed_lens_light.append({})
    kwargs_lens_light_init.append({'R_sersic': radpix_secondblob*deltaPix, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': x0blob*deltaPix, 'center_y': (60-y0blob)*deltaPix})
    kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.3, 'e1': 0.05, 'e2': 0.05, 'center_x': 0.05, 'center_y': 0.05})
    kwargs_lower_lens_light.append({'e1': -0.3, 'e2': -0.3, 'R_sersic': 3*deltaPix, 'n_sersic': 0.5, 'center_x': (x0blob-4)*deltaPix, 'center_y': ((60-y0blob)-4)*deltaPix})
    kwargs_upper_lens_light.append({'e1': 0.1, 'e2': 0.1, 'R_sersic': 10*deltaPix, 'n_sersic': 6, 'center_x': (x0blob+4)*deltaPix, 'center_y': ((60-y0blob)+4)*deltaPix})

    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light,
                         kwargs_upper_lens_light]

    kwargs_params = {'lens_light_model': lens_light_params}

    from lenstronomy.Workflow.fitting_sequence import FittingSequence

    fitting_seqI = FittingSequence(multi_band_listI, kwargs_modelI, kwargs_constraints, kwargs_likelihood,
                                   kwargs_params)

    fitting_kwargs_list = [['MCMC', {'n_burn': 100, 'n_run': 100, 'walkerRatio': 10,'sigma_scale': .1}]]

    chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seqI.fit_sequence(fitting_kwargs_list)
    lens_result, source_result, lens_light_resultI, ps_result, cosmo_result = fitting_seqI.best_fit()

    fitting_seqR = FittingSequence(multi_band_listR, kwargs_modelR, kwargs_constraints, kwargs_likelihood,
                                   kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 180, 'n_iterations': 180}]]

    chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seqR.fit_sequence(fitting_kwargs_list)
    lens_result, source_result, lens_light_resultR, ps_result, cosmo_result = fitting_seqR.best_fit()

    fitting_seqG = FittingSequence(multi_band_listG, kwargs_modelG, kwargs_constraints, kwargs_likelihood,
                                   kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 180, 'n_iterations': 180}]]

    chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seqG.fit_sequence(fitting_kwargs_list)
    lens_result, source_result, lens_light_resultG, ps_result, cosmo_result = fitting_seqG.best_fit()



    print 'end fit', lens_light_resultI
    np.save('.\\fit_param_2sersics\\'+nameI+'lens_light_resultI.npy', kwargs_dataI)
    np.save('.\\fit_param_2sersics\\' + nameG + 'lens_light_resultG.npy', kwargs_dataG)
    np.save('.\\fit_param_2sersics\\' + name + 'lens_light_resultR.npy', kwargs_dataR)

    lensPlotI = LensModelPlot(kwargs_dataI, kwargs_psfI, kwargs_numerics, kwargs_modelI, lens_result, source_result,
                                 lens_light_resultI, ps_result, arrow_size=0.02, cmap_string="gist_heat")

    lensPlotR = LensModelPlot(kwargs_dataR, kwargs_psfR, kwargs_numerics, kwargs_modelR, lens_result, source_result,
                              lens_light_resultR, ps_result, arrow_size=0.02, cmap_string="gist_heat")
    lensPlotG = LensModelPlot(kwargs_dataG, kwargs_psfG, kwargs_numerics, kwargs_modelG, lens_result, source_result,
                              lens_light_resultG, ps_result, arrow_size=0.02, cmap_string="gist_heat")

    for i in range(len(chain_list)):
            if len(param_list[i]) > 0:
                f, axes = out_plot.plot_chain(chain_list[i], param_list[i])

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)
    plt.title('I band')
    lensPlotI.data_plot(ax=axes[0])
    lensPlotI.model_plot(ax=axes[1])
    lensPlotI.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)

    #plt.show()
    plt.savefig('E:\\twosersicresults\\'+name[0:8]+'Ibandplot2.png')

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)
    plt.title('R band')
    lensPlotR.data_plot(ax=axes[0])
    lensPlotR.model_plot(ax=axes[1])
    lensPlotR.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.savefig('E:\\twosersicresults\\' + name[0:8] + 'Rbandplot2.png')
    #plt.show()

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)
    plt.title('G band')
    lensPlotG.data_plot(ax=axes[0])
    lensPlotG.model_plot(ax=axes[1])
    lensPlotG.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6)
    f.tight_layout()

    plt.savefig('E:\\twosersicresults\\' + name[0:8] + 'Gbandplot2.png')
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    #plt.show()






    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotI.decomposition_plot(ax=axes[0], text='Lens light I', lens_light_add=True, unconvolved=True)
    lensPlotI.decomposition_plot(ax=axes[1], text='Lens light convolved I', lens_light_add=True)
    lensPlotI.subtract_from_data_plot(ax=axes[2], text='Data - Lens  I', lens_light_add=True)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Ibandplot3.png')
    #plt.show()
    #print(lens_light_result)

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotR.decomposition_plot(ax=axes[0], text='Lens light R', lens_light_add=True, unconvolved=True)
    lensPlotR.decomposition_plot(ax=axes[1], text='Lens light convolved R', lens_light_add=True)
    lensPlotR.subtract_from_data_plot(ax=axes[2], text='Data - Lens  R', lens_light_add=True)
    f.tight_layout()
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Rbandplot3.png')
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    #plt.show()


    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotG.decomposition_plot(ax=axes[0], text='Lens light G', lens_light_add=True, unconvolved=True)
    lensPlotG.decomposition_plot(ax=axes[1], text='Lens light convolved G', lens_light_add=True)
    lensPlotG.subtract_from_data_plot(ax=axes[2], text='Data - Lens  G', lens_light_add=True)
    f.tight_layout()
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Ibandplot3.png')

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotI.decomposition_plot(ax=axes[0], text='Lens light I', lens_light_add=True, unconvolved=True)
    lensPlotI.decomposition_plot(ax=axes[1], text='Lens light convolved I', lens_light_add=True)
    lensPlotI.subtract_from_data_plot(ax=axes[2], text='Data - Lens  I', lens_light_add=True)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Ibandplot3.png')
    #plt.show()
    #print(lens_light_result)

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotR.decomposition_plot(ax=axes[0], text='Lens light R', lens_light_add=True, unconvolved=True)
    lensPlotR.decomposition_plot(ax=axes[1], text='Lens light convolved R', lens_light_add=True)
    lensPlotR.subtract_from_data_plot(ax=axes[2], text='Data - Lens  R', lens_light_add=True)
    f.tight_layout()
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Rbandplot3.png')
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    #plt.show()


    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlotG.decomposition_plot(ax=axes[0], text='Lens light G', lens_light_add=True, unconvolved=True)
    lensPlotG.decomposition_plot(ax=axes[1], text='Lens light convolved G', lens_light_add=True)
    lensPlotG.subtract_from_data_plot(ax=axes[2], text='Data - Lens  G', lens_light_add=True)
    f.tight_layout()
    plt.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Ibandplot3.png')
    plt.close('all')
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=s.savefig('E:\\plotfittingsersic\\' + name[0:8] + 'Gbandplot3.png')
    #plt.show()
