import datetime

import numpy as np

import matplotlib.pyplot as plt

import astropy.io.fits as pyfits

from lenstronomy.Workflow.fitting_sequence import FittingSequence

from lenstronomy.Plots.output_plots import LensModelPlot

from lenstronomy.Util import util

# %matplotlib inline




fwhm=10
num_images = 4

num_pix = 99

delta_pix = 0.08

image_data = pyfits.open('E:\\hSCsources\\arch-190206-122459\\20001670-778-cutout-HSC-I-9813-pdr1_udeep.fits')[0].data

#noise_map = pyfits.open('noise_map.fits')[0].data

#psf_kernel = pyfits.open('psf.fits')[0].data

_, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
 \
    = util.make_grid_with_coordtransform(numPix=num_pix,

                                         deltapix=delta_pix,

                                         subgrid_res=1,

                                         inverse=False)  # !!
#'noise_map': noise_map,
kwargs_data = {

    'image_data': image_data,
    'exp_time': 100,
    'background_rms': 0.05,

    'ra_at_xy_0': ra_at_xy_0,

    'dec_at_xy_0': dec_at_xy_0,

    'transform_pix2angle': Mpix2coord

}

kwargs_psf = {

    #'psf_type': 'PIXEL',
    'psf_type': 'GAUSSIAN',
    'fwhm': fwhm,
    'kernelsize':31,

    #'kernel_point_source': psf_kernel,

}

kwargs_numerics = {

    'subgrid_res': 2,

    'point_source_subgrid': 3,

    'psf_subgrid': 1,

    'subsampling_size': 11,

    # 'fix_psf_error_map': False,

    # 'mask': None,

    # 'idex_mask': None,

}

kwargs_model = {

    'lens_model_list': ['SPEMD', 'SHEAR'],

    'source_light_model_list': ['SERSIC_ELLIPSE'],

    'lens_light_model_list': ['SERSIC_ELLIPSE'],

    'point_source_model_list': ['LENSED_POSITION'],

    'fixed_magnification_list': [False]

}

# - initial parameter values

kwargs_shear_init = {'e1': 0., 'e2': 0.}

kwargs_spemd_init = {'theta_E': 1.,

                     'gamma': 2.,

                     'center_x': 0.,

                     'center_y': 0.,

                     'e1': 0.,

                     'e2': 0.}

kwargs_lens_init = [kwargs_spemd_init, kwargs_shear_init]

kwargs_sersic_ellipse_source_init = {'R_sersic': 0.3,

                                     'n_sersic': 3.,

                                     'center_x': 0.,

                                     'center_y': 0.,

                                     'e1': 0.,

                                     'e2': 0.}

kwargs_source_light_init = [kwargs_sersic_ellipse_source_init]

kwargs_sersic_ellipse_lens_init = {'R_sersic': 1.,

                                   'n_sersic': 4.,

                                   'center_x': 0.,

                                   'center_y': 0.,

                                   'e1': 0.,

                                   'e2': 0.}

kwargs_lens_light_init = [kwargs_sersic_ellipse_lens_init]

# - initial spread in parameter estimation

kwargs_shear_sigma = {'e1': 0.2, 'e2': 0.2}

kwargs_spemd_sigma = {'theta_E': 0.3,

                      'e1': 0.2,

                      'e2': 0.2,

                      'gamma': 0.05,

                      'center_x': 0.2,

                      'center_y': 0.2}

kwargs_lens_sigma = [kwargs_spemd_sigma, kwargs_shear_sigma]

kwargs_sersic_ellipse_source_sigma = {'R_sersic': 0.3,

                                      'n_sersic': 0.5,

                                      'e1': 0.2,

                                      'e2': 0.2,

                                      'center_x': 0.2,

                                      'center_y': 0.2}

kwargs_source_light_sigma = [kwargs_sersic_ellipse_source_sigma]

kwargs_sersic_ellipse_lens_sigma = {'R_sersic': 0.3,

                                    'n_sersic': 0.5,

                                    'e1': 0.2,

                                    'e2': 0.2,

                                    'center_x': 0.2,

                                    'center_y': 0.2}

kwargs_lens_light_sigma = [kwargs_sersic_ellipse_lens_sigma]

# - hard bound lower limit in parameter space

kwargs_shear_lower = {'e1': -0.3, 'e2': -0.3}

kwargs_spemd_lower = {'theta_E': 0.,

                      'gamma': 1.8,

                      'center_x': -5.,

                      'center_y': -5.,

                      'e1': -0.5,

                      'e2': -0.5}

kwargs_lens_lower = [kwargs_spemd_lower, kwargs_shear_lower]

kwargs_sersic_ellipse_source_lower = {'R_sersic': 0.001,

                                      'n_sersic': 0.5,

                                      'center_x': -5.,

                                      'center_y': -5.,

                                      'e1': -0.5,

                                      'e2': -0.5}

kwargs_source_light_lower = [kwargs_sersic_ellipse_source_lower]

kwargs_sersic_ellipse_lens_lower = {'R_sersic': 0.001,

                                    'n_sersic': 0.5,

                                    'center_x': -1.,

                                    'center_y': -1.,

                                    'e1': -0.5,

                                    'e2': -0.5}

kwargs_lens_light_lower = [kwargs_sersic_ellipse_lens_lower]

# - hard bound upper limit in parameter space

kwargs_shear_upper = {'e1': 0.3, 'e2': 0.3}

kwargs_spemd_upper = {'theta_E': 6.,

                      'gamma': 2.2,

                      'center_x': 1.,

                      'center_y': 1.,

                      'e1': 0.5,

                      'e2': 0.5}

kwargs_lens_upper = [kwargs_spemd_upper, kwargs_shear_upper]

kwargs_sersic_ellipse_source_upper = {'R_sersic': 5.,

                                      'n_sersic': 7.,

                                      'center_x': 3.,

                                      'center_y': 3.,

                                      'e1': 0.5,

                                      'e2': 0.5}

kwargs_source_light_upper = [kwargs_sersic_ellipse_source_upper]

kwargs_sersic_ellipse_lens_upper = {'R_sersic': 5.,

                                    'n_sersic': 7.,

                                    'center_x': 1.,

                                    'center_y': 1.,

                                    'e1': 0.5,

                                    'e2': 0.5}

kwargs_lens_light_upper = [kwargs_sersic_ellipse_lens_upper]

lens_params = [kwargs_lens_init,

               kwargs_lens_sigma,

               [{}, {}],

               kwargs_lens_lower,

               kwargs_lens_upper]

source_light_params = [kwargs_source_light_init,

                       kwargs_source_light_sigma,

                       [{}],

                       kwargs_source_light_lower,

                       kwargs_source_light_upper]

lens_light_params = [kwargs_lens_light_init,

                     kwargs_lens_light_sigma,

                     [{}],

                     kwargs_lens_light_lower,

                     kwargs_lens_light_upper]

kwargs_ps_init = [{'ra_image': np.array([0.2, -1.2, 1.3, -0.2]),

                   'dec_image': np.array([-1.1, -0.2, 0.3, 1.])}]

kwargs_ps_sigma = [{'ra_image': [0.05] * num_images,

                    'dec_image': [0.05] * num_images}]

kwargs_ps_lower = [{'ra_image': [-3.] * num_images,

                    'dec_image': [-3.] * num_images}]

kwargs_ps_upper = [{'ra_image': [3.] * num_images,

                    'dec_image': [3.] * num_images}]

point_source_params = [kwargs_ps_init, kwargs_ps_sigma, [{}],

                       kwargs_ps_lower, kwargs_ps_upper]

# cosmo_params = [{}] * 5

kwargs_params = {

    'lens_model': lens_params,

    'source_model': source_light_params,

    'lens_light_model': lens_light_params,

    'point_source_model': point_source_params,

    # 'cosmography': cosmo_params,

}

align_centers = True

joint_src_and_ps = [[0, 0]] if align_centers else []

ns = len(kwargs_model['source_light_model_list'])

kwargs_constraints = {

    'joint_source_with_point_source': joint_src_and_ps,

    'additional_images_list': [False],

    'image_plane_source_list': [False] * ns,

    'num_point_source_list': [num_images],

    'solver': True,

    'solver_type': 'PROFILE_SHEAR'

}

kwargs_likelihood = {

    'check_bounds': True,

    'force_no_add_image': False,

    'source_marg': False,

    'image_likelihood': True,

    'point_source_likelihood': False,

    'position_uncertainty': 0.004,

    'check_solver': True,

    'solver_tolerance': 0.001,

    # 'check_positive_flux': True,

}

image_band = [kwargs_data, kwargs_psf, kwargs_numerics]

multi_band_list = [image_band]

fittingSeq = FittingSequence(multi_band_list,

                             kwargs_model,

                             kwargs_constraints,

                             kwargs_likelihood,

                             kwargs_params)

kwargs_pso = {

    'fitting_routine': 'PSO',

    'mpi': False,

    'n_particles': 50,

    'n_iterations': 100,

    'sigma_scale': 1,

}
'''
fitting_kwargs_list = [kwargs_pso]

lens_result, source_light_result, lens_light_result, ps_result, \
 \
cosmo_result, chain_list, param_list, \
 \
samples_mcmc, param_mcmc, dist_mcmc \
 \
    = fittingSeq.fit_sequence(fitting_kwargs_list)

now = datetime.datetime.now()

unique_id = now.strftime('%y%m%d_%H%M')

lensPlot = LensModelPlot(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model,

                         lens_result, source_light_result, lens_light_result, ps_result,

                         arrow_size=0.02, cmap_string="gist_heat")

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

lensPlot.data_plot(ax=axes[0, 0])

lensPlot.model_plot(ax=axes[0, 1])

lensPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)

lensPlot.source_plot(ax=axes[1, 0], convolution=False, deltaPix_source=0.01, numPix=100)

lensPlot.convergence_plot(ax=axes[1, 1], v_max=1)

lensPlot.magnification_plot(ax=axes[1, 2])

f.tight_layout()

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)

f.savefig('result1_{}.png'.format(unique_id), format='png')

f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

lensPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)

lensPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)

lensPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)

lensPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)

lensPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                            unconvolved=True)

lensPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True, lens_light_add=True,
                            point_source_add=True)

f.tight_layout()

f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)

#f.savefig('result2_{}.png'.format(unique_id), format='png')



# plt.show()
'''