import numpy as np

import jax
import jax.numpy as jnp
from jax_zero_contour import ZeroSolver
from jax import random

from functools import partial

import matplotlib.pyplot as plt

from skimage import measure
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev

import numpyro
import numpyro.distributions as dist

from astropy.io import fits

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage

import os


class EuclidGenerator:

    PIXEL_SCALE = 0.1       # arseconds/pixel
    BACKGROUND_RMS = 4.2    # e- rms
    PSF_FWHM = 0.16         # arcseconds

    THETA_PDF_FILEPATH = "euclid_generator/data/theta_pdf.npy"
    ETA_PDF_FILEPATH = "euclid_generator/data/eta_pdf.npy"

    def __init__(self, lens_folder_path: str, fov: float = 15.0, *, exposure_time : float = 600.0, lens_amp: float = 1.0, source1_amp: float = 1.0, source2_amp:float = 1.0) -> None:
        """
        Initialise a generator class. This class with generate a number of images for euclid dspl discovery
        """

        self.fov = fov
        self.exposure_time = exposure_time
        self.lens_amp = lens_amp
        self.source1_amp = source1_amp
        self.source2_amp = source2_amp

        self.lens_folder_path = lens_folder_path
        with os.scandir(lens_folder_path) as entries:
            self.lens_files = np.array([item.name for item in entries])
        self.n_sims = len(self.lens_files)

        self.pixel_grid = self.__initialise_pixel_grid()
        self.noise = self.__initialise_noise()
        self.psf = self.__initialise_psf()

        self.mass_model, self.light_model = self.__initialise_models()
        self.LensImage = self.__initialise_LensImage()

        self.mass_kwargs = self.light_kwargs = None

    def __initialise_pixel_grid(self) -> PixelGrid:
        """
        initialise a square pixel grid instance

        Parameters
        ----------
        fov : float
            the field of view of the pixel grid in arcseconds
        pixel_scale : float
            the pixel width of the grid in arcseconds
        """
        pixel_scale = self.PIXEL_SCALE
        fov = self.fov

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
    
    def __initialise_noise(self) -> Noise:
        nx, ny = self.pixel_grid.num_pixel_axes
        return Noise(nx, ny, background_rms=self.BACKGROUND_RMS, exposure_time=self.exposure_time)

    def __initialise_psf(self) -> PSF:
        return PSF(psf_type='GAUSSIAN', fwhm=self.PSF_FWHM, pixel_size=self.PIXEL_SCALE)

    def __initialise_light_pixelated(self, pixels: np.ndarray) -> LightModel:
        """
        if self.lens_img is an array, initialises and returns a LightModel(['PIXELATED']) instance 
        """
        assert isinstance(pixels, np.ndarray)
        ny_img, nx_img = pixels.shape

        px_scale_x = float(self.fov) / float(nx_img)
        px_scale_y = float(self.fov) / float(ny_img)
        px_scale_img = 0.5 * (px_scale_x + px_scale_y)

        position = (0, 0)
        light_kwargs = {
            'grid_center': position,
            'grid_shape': (self.fov, self.fov),
            'pixel_scale_factor': px_scale_img / self.PIXEL_SCALE
        }
        light_model = LightModel(['PIXELATED'], pixel_interpol='bilinear', kwargs_pixelated=light_kwargs) 
        return light_model

    def __initialise_models(self) -> tuple[MPMassModel, MPLightModel]:
        """initialise mass and light models of all planes in system"""

        # --- mass --- #
        lens_mass_model = MassModel(['EPL', 'SHEAR'])
        source1_mass_model = MassModel(['SIS', 'SHEAR'])

        mp_mass_model = MPMassModel([
            lens_mass_model,
            source1_mass_model
        ])

        # --- light --- #
        with fits.open(self.lens_folder_path + self.lens_files[0]) as hdul:
            img = hdul[1].data
        lens_light_model = self.__initialise_light_pixelated(img)

        source1_light_model = LightModel(['SERSIC_ELLIPSE', 'MULTIPOLE', 'MULTIPOLE'])

        source2_light_model = LightModel(['SERSIC_ELLIPSE', 'MULTIPOLE', 'MULTIPOLE'])


        mp_light_model = MPLightModel([
            lens_light_model,       # lens galaxy light
            source1_light_model,    # first source light
            source2_light_model     # second source light
        ])

        return mp_mass_model, mp_light_model

    def __initialise_LensImage(self):
        lens_image = MPLensImage(
            grid_class=self.pixel_grid,
            psf_class=self.psf,
            noise_class=self.noise,
            mass_model_class=self.mass_model,
            light_model_class=self.light_model
        )
        return lens_image

    """generator methods"""

    @staticmethod
    def __compute_ellipticity(q, phi):
        e = (1 - q) / (1 + q)
        e1 = e * np.cos(2 * phi)
        e2 = e * np.sin(2 * phi)
        return e1, e2
        
    @staticmethod
    def __sample_from_pdf(pdf_filepath):
        data = jnp.load(pdf_filepath)
        x, pdf = data[:, 0], data[:, 1]

        dx = jnp.diff(x, append=x[-1])  
        cdf = jnp.cumsum(pdf * dx)
        cdf /= cdf[-1]  

        val = np.random.rand()
        samples = jnp.interp(val, cdf, x)

        return samples

    def __sample_kwargs(self, key, image_index):
        
        # --- lens --- #
        e1, e2 = EuclidGenerator.__compute_ellipticity(
            q = random.uniform(key, minval=0.2, maxval=1.0), 
            phi = random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        )

        lens_EPL_kwargs = {
            'theta_E' : jnp.float32(EuclidGenerator.__sample_from_pdf(EuclidGenerator.THETA_PDF_FILEPATH)), 
            'gamma': jnp.float32(random.uniform(key=key, minval=1.7, maxval=2.3)),
            'e1': jnp.float32(e1),
            'e2': jnp.float32(e2),
            'center_x': jnp.float32(0.0),
            'center_y': jnp.float32(0.0)
        }
        lens_shear_kwargs = {
            'gamma1': jnp.float32(random.uniform(key=key, minval=0.0, maxval=0.3)),     
            'gamma2': jnp.float32(random.uniform(key=key, minval=0.0, maxval=0.3)),      
            'ra_0': jnp.float32(lens_EPL_kwargs['center_x']),
            'dec_0': jnp.float32(lens_EPL_kwargs['center_y']),
        }
        lens_mass_kwargs = [lens_EPL_kwargs, lens_shear_kwargs]
        

        with fits.open(self.lens_folder_path + self.lens_files[image_index]) as hdul:
            img = jnp.asarray(hdul[1].data, dtype=jnp.float32)

        lens_light_kwargs = [{
            "pixels" : self.lens_amp * img
        }]
        
        # --- fake source 1 kwargs --- #
        source1_SIS_kwargs = {
            'theta_E' : jnp.float32(0.0), 
            'center_x': jnp.float32(0.0),    
            'center_y': jnp.float32(0.0)    
        }
        source1_shear_kwargs = {
            'gamma1': jnp.float32(0.0),     
            'gamma2': jnp.float32(0.0),      
            'ra_0': jnp.float32(0.0),
            'dec_0': jnp.float32(0.0),
        }
        source1_mass_kwargs = [source1_SIS_kwargs, source1_shear_kwargs]

        mp_mass_kwargs = [lens_mass_kwargs, source1_mass_kwargs]
        eta = jnp.float32(self.__sample_from_pdf(EuclidGenerator.ETA_PDF_FILEPATH))

        # compute bounds on s1, s2 position with tangenial caustic
        # TODO can probably make this better
        tangential_caustic = np.array(self.get_tangential_caustic(mp_mass_kwargs, eta))
        xmin = jnp.min(tangential_caustic[:, 0])
        xmax = jnp.max(tangential_caustic[:, 0])
        ymin = jnp.min(tangential_caustic[:, 1])
        ymax = jnp.max(tangential_caustic[:, 1])
        
        source1_SIS_kwargs = {
            'theta_E' : jnp.float32(EuclidGenerator.__sample_from_pdf(EuclidGenerator.THETA_PDF_FILEPATH)), 
            'center_x': jnp.float32(random.uniform(key, minval=xmin, maxval=xmax)),    
            'center_y': jnp.float32(random.uniform(key, minval=ymin, maxval=ymax))  
        }
        source1_shear_kwargs = {
            'gamma1': jnp.float32(random.uniform(key=key, minval=0.0, maxval=0.3)),     
            'gamma2': jnp.float32(random.uniform(key=key, minval=0.0, maxval=0.3)),         
            'ra_0': jnp.float32(source1_SIS_kwargs['center_x']),
            'dec_0': jnp.float32(source1_SIS_kwargs['center_y']),
        }

        e1, e2 = EuclidGenerator.__compute_ellipticity(
            q = random.uniform(key, minval=0.2, maxval=1.0), 
            phi = random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        )

        source1_SERSIC_kwargs = {
            'amp': jnp.float32(self.source1_amp),
            'R_sersic': jnp.float32(0.4),    # TODO
            'n_sersic': jnp.float32(4.0),    # TODO 
            'e1': jnp.float32(e1),          # TODO
            'e2': jnp.float32(e2),          # TODO
            'center_x': jnp.float32(source1_SIS_kwargs['center_x']), 
            'center_y': jnp.float32(source1_SIS_kwargs['center_y'])
        }
        source1_MULTIPOLE3_kwargs = {
            'm' : 3,
            "amp_m" : 0.2,
            "phi_m" : jnp.pi/4,
            'center_x' : 2.0,
            'center_y' : 3.0            
        }
        source1_MULTIPOLE4_kwargs = {
            'm' : 4,
            "amp_m" : 0.2,
            "phi_m" : jnp.pi/4,
            'center_x' : 2.0,
            'center_y' : 3.0            
        }

        source1_light_kwargs = [source1_SERSIC_kwargs, source1_MULTIPOLE3_kwargs, source1_MULTIPOLE4_kwargs]

        # --- source 2 --- #
        e1, e2 = EuclidGenerator.__compute_ellipticity(
            q = random.uniform(key, minval=0.2, maxval=1.0), 
            phi = random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        )

        source2_SERSIC_kwargs = {
            'amp': jnp.float32(self.source1_amp),
            'R_sersic': jnp.float32(0.4),    # TODO
            'n_sersic': jnp.float32(4.0),    # TODO 
            'e1': jnp.float32(e1),          # TODO
            'e2': jnp.float32(e2),          # TODO
            'center_x': jnp.float32(source1_SIS_kwargs['center_x']), 
            'center_y': jnp.float32(source1_SIS_kwargs['center_y'])
        }
        source2_MULTIPOLE3_kwargs = {
            'm' : 3,
            "amp_m" : 0.2,
            "phi_m" : jnp.pi/4,
            'center_x' : 2.0,
            'center_y' : 3.0            
        }
        source2_MULTIPOLE4_kwargs = {
            'm' : 4,
            "amp_m" : 0.2,
            "phi_m" : jnp.pi/4,
            'center_x' : 2.0,
            'center_y' : 3.0            
        }
        source2_light_kwargs = [source2_SERSIC_kwargs, source2_MULTIPOLE3_kwargs, source2_MULTIPOLE4_kwargs]

        mp_light_kwargs = [lens_light_kwargs, source1_light_kwargs, source2_light_kwargs]

        return mp_mass_kwargs, mp_light_kwargs, eta



    def get_inverse_magnification(self, x, y, mass_kwargs, eta, plane):
        A = self.mass_model.A( x=x, y=y, kwargs=mass_kwargs, eta_flat=eta )
        Ap = A[plane, :, :]
        return Ap[..., 0, 0] * Ap[..., 1, 1] - Ap[..., 0, 1] * Ap[..., 1, 0]

    def get_critical_curves(self, mass_kwargs, eta, plane=1):

        x_vec = jnp.linspace(-20.0, 20.0, 100)
        y_vec = jnp.linspace(-20.0, 20.0, 100)
        X, Y = jnp.meshgrid(x_vec, y_vec)

        mu_inv = self.get_inverse_magnification(
            X, Y, mass_kwargs, eta, plane
        )

        mu_inv_np = np.asarray(mu_inv)

        contours = measure.find_contours(mu_inv_np, level=0.0)

        critical_curves = []

        for contour in contours:
            iy, ix = contour.T

            x = x_vec[ix.astype(int)]
            y = y_vec[iy.astype(int)]

            critical_curves.append((x, y))

        return critical_curves
    
    def get_caustics(self, mass_kwargs, eta, plane=1):
        critical_curves = self.get_critical_curves(mass_kwargs, eta, plane)
        
        caustics = []
        for curve in critical_curves:
            x_img = jnp.array(curve)[:, 0]
            y_img = jnp.array(curve)[:, 0]

            x_def, y_def = self.mass_model.ray_shooting(x_img, y_img, eta_flat=eta, kwargs=mass_kwargs)

            x_s1 = x_def.T[:, 1]
            y_s1 = y_def.T[:, 1]

            caustics.append(np.column_stack([np.array(x_s1), np.array(y_s1)]))

        return caustics
    
    def get_tangential_caustic(self, mass_kwargs, eta, plane=1):
        caustics = self.get_caustics(mass_kwargs, eta, plane)

        if not caustics:
            return None

        return max(caustics, key=lambda curve: len(curve[0]))

    def get_simulation(self, mass_kwargs, light_kwargs, eta, noise_seed) -> np.ndarray:
        
        sim = self.LensImage.simulation(
            kwargs_mass=mass_kwargs,
            kwargs_light=light_kwargs,
            eta_flat=eta,
            noise_seed=random.PRNGKey(noise_seed)
        )

        return sim

    @staticmethod
    def main():
        gen = EuclidGenerator("euclid_generator/lrg_used_for_karina_sim/", 17, lens_amp=100, source1_amp=100, source2_amp=100)
        mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(random.PRNGKey(0), 0)
        sim = gen.get_simulation(mass_kwargs, light_kwargs, eta, 0)

        sim_np = np.asarray(sim)
        plt.imshow(sim_np, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.savefig("euclid_generator/images/testing.png")

EuclidGenerator.main()
