
import numpy as np
import jax.numpy as jnp
from jax import random

# import pandas as pd

# import scipy as sp
# from scipy.integrate import quad
# from scipy.stats import multivariate_normal
# import matplotlib.pyplot as plt

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image import LensImage

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage


class DSPL:


    @staticmethod
    def initialise_pixel_grid(fov: float, pixel_scale: float) -> PixelGrid:
        """
        initialise a square pixel grid instance

        Parameters
        ----------
        fov : float
            the field of view of the pixel grid in arcseconds
        pixel_scale : float
            the pixel width of the grid in arcseconds
        """

        nx = int(np.floor(fov/pixel_scale))
        ny = int(np.floor(fov/pixel_scale))

        # set central pixel to (0,0)
        half_size_x = (nx - 1) / 2
        half_size_y = (ny - 1) / 2
        ra_at_xy_0 = - half_size_x * pixel_scale
        dec_at_xy_0 = - half_size_y * pixel_scale

        transform_pix2angle = pixel_scale*np.eye(2)
        kwargs_pixel = {
            'nx': nx,
            'ny': ny,
            'ra_at_xy_0': ra_at_xy_0,
            'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': transform_pix2angle
        }

        return PixelGrid(**kwargs_pixel)

    PIXEL_SCALE = {'hubble': 0.04, 'euclid': 0.1, 'JWST': 0.063} # arcseconds
    BACKGROUND_RMS = {'hubble': 8.7, 'euclid': 4.2, 'JWST': 15.8} # e- rms 
    PSF_FILE = {# telescope : (filename, supersampling factor)
        'hubble': ('hubble_psf.npy', 4), 
        'euclid': ('euclid_psf.npy', 1),
        'JWST': ('JWST_psf.npy', 7)
    }
    
    def __init__(self, 
                 *, 
                 fov: float = 2.0, 
                 telescope: str ='euclid', 
                 exposure_time: float = 600.0,
                 jaxRNG_key = random.PRNGKey(18)
                 ) -> None:
        
        # intialise pixel grid to view everything
        pixel_grid = self.initialise_pixel_grid(fov, pixel_scale=self.PIXEL_SCALE[telescope])

        nx, ny = pixel_grid.num_pixel_axes
        xgrid, ygrid = pixel_grid.pixel_coordinates
        x_axis = xgrid[0]
        y_axis = ygrid[:, 0]
        pixel_width = pixel_grid.pixel_width
        extent = pixel_grid.extent

        # initialise noise, psf and eta
        noise = Noise(nx, ny, background_rms=self.BACKGROUND_RMS[telescope], exposure_time=exposure_time)

        psf_filename, supersampling_factor = self.PSF_FILE[telescope]
        psf = PSF(
            psf_type='PIXEL',
            pixel_size=pixel_width,
            kernel_point_source=np.load(f"psf/{psf_filename}"),
            kernel_supersampling_factor=supersampling_factor
        )

        eta = random.uniform(key=jaxRNG_key, minval=1.0, maxval=5.0) # TODO make this match the q1 data

        # initialise mass and light models and place in MP_Model containers
        lens_mass_model = MassModel(['EPL', 'SHEAR']) 
        lens_light_model = LightModel(['SERSIC_ELLIPSE'])

        source1_mass_model = MassModel(['SIS', 'SHEAR']) 
        source1_light_model = LightModel(['SERSIC_ELLIPSE'])  

        source2_light_model = LightModel(['SERSIC_ELLIPSE'])

        mp_mass_model = MPMassModel([
        lens_mass_model,          # deflector
        source1_mass_model,       # source 1 mass model
        None                      # no mass at source 2 plane
        ])

        mp_light_model = MPLightModel([
            lens_light_model,     # lens galaxy light
            source1_light_model,  # first source light
            source2_light_model   # second source light
        ])

        # set mass parameters
        lens_EPL_kwargs = {
            'theta_E': random.uniform(key=jaxRNG_key, minval=0.5, maxval=1.0), # NOTE might change this distribution
            'gamma': 2.05, # TODO 
            'e1': random.uniform(key=jaxRNG_key, minval=0.0, maxval=1.0),
            'e2': random.uniform(key=jaxRNG_key, minval=0.0, maxval=1.0),
            'center_x': 0.0,
            'center_y': 0.0
        }
        lens_shear_kwargs = {
            'gamma1': 0.0,
            'gamma2': 0.0,
            'ra_0': lens_EPL_kwargs['center_x'],
            'dec_0': lens_EPL_kwargs['center_y'],
        }

        source1_SIS_kwargs = {
            'theta_E': 0.1, # TODO 
            'center_x': 0.0,
            'center_y': 0.0
        }
        source1_shear_kwargs = {
            'gamma1': 0.0,
            'gamma2': -0.03,
            'ra_0': source1_SIS_kwargs['center_x'],
            'dec_0': source1_SIS_kwargs['center_y'],
        }

        
        
        

