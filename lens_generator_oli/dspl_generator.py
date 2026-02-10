import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from skimage import measure
from scipy.interpolate import splprep, splev

import numpyro
import numpyro.distributions as dist

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage


"""
TODO
    - need to find a way to set brightness
    - need to fit a profile to pixel images
"""


class DSPLGenerator():

    PIXEL_SCALE = {'hubble': 0.04, 'euclid': 0.1, 'JWST': 0.063, 'example': 0.005} # arcseconds
    BACKGROUND_RMS = {'hubble': 8.7e-2, 'euclid': 4.2e-2, 'JWST': 15.8e-2, 'example': 0} # me- rms 
    PSF_FILE = {# telescope : (filename, supersampling factor)
        'hubble': ('hubble_psf.npy', 4), 
        'euclid': ('euclid_psf.npy', 1),
        'JWST': ('JWST_psf.npy', 7),
        'example' : ('hubble_psf.npy', 4)
    }
    
    """initialiser methods"""
    

    def __init__(self, 
                 fov: float, 
                 lens_img: str | np.ndarray | None = None,
                 source1_img: str | np.ndarray | None = None,
                 source2_img: str | np.ndarray | None = None,
        
                 telescope: str = 'euclid',
                 exposure_time: float = 6e2,
                 lens_amp=1.0,
                 source1_amp=1.0,
                 source2_amp=1.0
                 ) -> None:
        
        self.fov = fov
        self.exposure_time = exposure_time
        self.telescope = telescope
        self.lens_amp = lens_amp
        self.source1_amp = source1_amp
        self.source2_amp = source2_amp

        # lens
        if isinstance(lens_img, str):
            self.lens_img = np.load(lens_img)
        elif isinstance(lens_img, np.ndarray):
            self.lens_img = lens_img
        else:
            self.lens_img = None

        # source 1
        if isinstance(source1_img, str):
            self.source1_img = np.load(source1_img)
        elif isinstance(source1_img, np.ndarray):
            self.source1_img = source1_img
        else:
            self.source1_img = None

        # source 2
        if isinstance(source2_img, str):
            self.source2_img = np.load(source2_img)
        elif isinstance(source2_img, np.ndarray):
            self.source2_img = source2_img
        else:
            self.source2_img = None

        self.pixel_grid = self._initialise_pixel_grid()
        self.mass_model, self.light_model = self._initialise_models()
        self.mass_kwargs, self.light_kwargs, self.eta = self._initialise_profile_parameters()
        self.noise, self.psf = self._initialise_noise_psf()
    
    # --- these are the same in Generator class --- #
    def _initialise_pixel_grid(self):
        """
        initialise a square pixel grid instance

        Parameters
        ----------
        fov : float
            the field of view of the pixel grid in arcseconds
        pixel_scale : float
            the pixel width of the grid in arcseconds
        """
        pixel_scale = self.PIXEL_SCALE[self.telescope]
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

    def _initialise_noise_psf(self) -> tuple[Noise, PSF]:
        """initialise noies and psf instances for the telescope"""
        nx, ny = self.pixel_grid.num_pixel_axes
        pixel_width = self.pixel_grid.pixel_width
        telescope = self.telescope
        exposure_time = self.exposure_time

        noise = Noise(nx, ny, background_rms=self.BACKGROUND_RMS[telescope], exposure_time=exposure_time)

        psf_filename, supersampling_factor = self.PSF_FILE[telescope]
        psf = PSF(
            psf_type='PIXEL',
            pixel_size=pixel_width,
            kernel_point_source=np.load(f"data/{psf_filename}"),
            kernel_supersampling_factor=supersampling_factor
        )
        return noise, psf

    # --- --- #

    def _initialise_light_pixelated(self, pixels: np.ndarray) -> LightModel:
        """
        if self.lens_img is an array, initialises and returns a LightModel(['PIXELTED']) instance 
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
            'pixel_scale_factor': px_scale_img / self.PIXEL_SCALE[self.telescope]
        }
        lens_light_model = LightModel(['PIXELATED'], pixel_interpol='bilinear', kwargs_pixelated=light_kwargs) 
        return lens_light_model

    def _initialise_models(self) -> tuple[MPMassModel, MPLightModel]:
        """initialise mass and light models of all planes in system"""

        # --- mass --- #
        lens_mass_model = MassModel(['EPL', 'SHEAR'])
        source1_mass_model = MassModel(['SIS', 'SHEAR'])

        mp_mass_model = MPMassModel([
            lens_mass_model,
            source1_mass_model
        ])

        # --- light --- #
        if isinstance(self.lens_img, np.ndarray):
            lens_light_model = self._initialise_light_pixelated(self.lens_img)
        else:
            lens_light_model = LightModel(["SERSIC_ELLIPSE"])

        if isinstance(self.source1_img, np.ndarray):
            source1_light_model = self._initialise_light_pixelated(self.source1_img)
        else:
            source1_light_model = LightModel(['SERSIC_ELLIPSE'])

        if isinstance(self.source2_img, np.ndarray):
            source2_light_model = self._initialise_light_pixelated(self.source2_img)
        else:
            source2_light_model = LightModel(['SERSIC_ELLIPSE'])


        mp_light_model = MPLightModel([
            lens_light_model,       # lens galaxy light
            source1_light_model,    # first source light
            source2_light_model     # second source light
        ])

        return mp_mass_model, mp_light_model
    

    ### --- This is where the random generation happens --- ###
    def _initialise_profile_parameters(self) -> tuple[list,list, float]: 
        """Initialiase mass and light profile kwargs
        TODO need to set brightness of profiles better and make it make more sense
        """
        # -- mass kwargs -- #
        def compute_ellipticity(q, phi):
            e = (1 - q) / (1 + q)
            e1 = e * np.cos(2 * phi)
            e2 = e * np.sin(2 * phi)
            return e1, e2
        
        e1, e2 = compute_ellipticity(
            q = np.random.uniform(0.2, 1.0), 
            phi = np.random.uniform(0, np.pi)
        )

        lens_EPL_kwargs = {
            'theta_E': dist.TruncatedNormal(loc=0.75, scale=0.25, low=0.0, high=2.0), 
            'gamma': np.random.uniform(1.7, 2.3),
            'e1': e1,
            'e2': e2,
            'center_x': 0.0,
            'center_y': 0.0
        }
        lens_shear_kwargs = {
            'gamma1': np.random.uniform(0.0, 0.05),     
            'gamma2': np.random.uniform(0.0, 0.05),      
            'ra_0': lens_EPL_kwargs['center_x'],
            'dec_0': lens_EPL_kwargs['center_y'],
        }

        source1_SIS_kwargs = {
            'theta_E': np.random.uniform(0.1, lens_EPL_kwargs['theta_E']),     
            'center_x': np.random.normal(0.0, lens_EPL_kwargs['theta_E']),    
            'center_y': np.random.normal(0.0, lens_EPL_kwargs['theta_E'])    
        }
        source1_shear_kwargs = {
            'gamma1': np.random.uniform(0.0, 0.05),     
            'gamma2': np.random.uniform(0.0, 0.05),       
            'ra_0': source1_SIS_kwargs['center_x'],
            'dec_0': source1_SIS_kwargs['center_y'],
        }

        lens_mass_kwargs = [lens_EPL_kwargs, lens_shear_kwargs]
        source1_mass_kwargs = [source1_SIS_kwargs, source1_shear_kwargs]

        mp_mass_kwargs = [lens_mass_kwargs, source1_mass_kwargs]
        
        # -- light kwargs -- #
        if "PIXELATED" in self.light_model.light_models[0].profile_type_list: # type: ignore
            assert isinstance(self.lens_img, np.ndarray)
            img = self.lens_amp * self.lens_img / np.max(self.lens_img)
            lens_light_kwargs = [{
                "pixels" : img
            }]
        elif "SERSIC_ELLIPSE" in self.light_model.light_models[0].profile_type_list: # type: ignore
            lens_light_kwargs = [{
                'amp': self.lens_amp,
                'R_sersic': lens_EPL_kwargs['theta_E'],
                'n_sersic': 4.0, # n=4: de Vaucouleurs profile
                'e1': lens_EPL_kwargs['e1'],
                'e2': lens_EPL_kwargs['e2'],
                'center_x': 0.0,
                'center_y': 0.0
            }]
        else:
            raise ValueError # NOTE for debugging - can remove later

        if "PIXELATED" in self.light_model.light_models[1].profile_type_list: # type: ignore
            assert isinstance(self.source1_img, np.ndarray)
            img = self.source1_amp * self.source1_img / np.max(self.source1_img)
            source1_light_kwargs = [{
                "pixels" : img
            }]
        elif "SERSIC_ELLIPSE" in self.light_model.light_models[1].profile_type_list: # type: ignore
           source1_light_kwargs = [{
                'amp': self.source1_amp,
                'R_sersic': lens_EPL_kwargs['theta_E'], # type: ignore
                'n_sersic': 4.0, # n=4: de Vaucouleurs profile
                'e1': 0.0,
                'e2': 0.0,
                'center_x': source1_SIS_kwargs['center_x'],
                'center_y': source1_SIS_kwargs['center_y']
            }]
        else:
            raise ValueError # NOTE for debugging - can remove later

        if "PIXELATED" in self.light_model.light_models[2].profile_type_list: # type: ignore
            assert isinstance(self.source2_img, np.ndarray)
            img = self.source2_amp * self.source2_img / np.max(self.source2_img)
            source2_light_kwargs = [{
                "pixels" : img
            }]
        elif "SERSIC_ELLIPSE" in self.light_model.light_models[2].profile_type_list: # type: ignore
            source2_light_kwargs = [{
                'amp': self.source2_amp,
                'R_sersic': lens_EPL_kwargs['theta_E']*1e-3, # TODO this is not very good
                'n_sersic': 4.0, # n=4: de Vaucouleurs profile
                'e1': 0.0,
                'e2': 0.0,
                'center_x': np.random.normal(0.0, lens_EPL_kwargs['theta_E']),
                'center_y': np.random.normal(0.0, lens_EPL_kwargs['theta_E'])
            }]
        else:
            raise ValueError # NOTE for debugging - can remove later

        mp_light_kwargs = [lens_light_kwargs, source1_light_kwargs, source2_light_kwargs]

        eta = np.random.uniform(1.0, 5.0) # TODO

        return mp_mass_kwargs, mp_light_kwargs, eta
        
    ### ---  --- ###

    """getter methods"""

    def get_lens_image(self) -> tuple[MPLensImage, list, list, jnp.ndarray]:
        """
        get the MPLensImage instance of this shuffle

        Returns
        -------
        lens_img : MPLensImage
            the multiplane lens image instance
        mass_kwargs: list
            A nested list of dictionaries which give the mass profile key word arguments
        light_kwargs: list
            A nested list of dictionaries which give the light profile key word arguments
        eta_flat : jnp.ndarray
            the eta value of the system within a jaxnumpy array
        """

        lens_image = MPLensImage(
            grid_class=self.pixel_grid,
            psf_class=self.psf,
            noise_class=self.noise,
            mass_model_class=self.mass_model,
            light_model_class=self.light_model
        )
        self.lens_image = lens_image
        return lens_image, self.mass_kwargs, self.light_kwargs, jnp.array([self.eta])
    
    def _get_lens_image(self) -> MPLensImage:
        """
        get the MPLensImage instance of this shuffle

        Returns
        -------
        lens_img : MPLensImage
            the multiplane lens image instance
        mass_kwargs: list
            A nested list of dictionaries which give the mass profile key word arguments
        light_kwargs: list
            A nested list of dictionaries which give the light profile key word arguments
        eta_flat : jnp.ndarray
            the eta value of the system within a jaxnumpy array
        """

        lens_image = MPLensImage(
            grid_class=self.pixel_grid,
            psf_class=self.psf,
            noise_class=self.noise,
            mass_model_class=self.mass_model,
            light_model_class=self.light_model
        )
        self.lens_image = lens_image
        return lens_image


    def get_model(self) -> np.ndarray:
        """
        initialises LensImage and returns an array of the lens model
        """
        if not hasattr(self, "lens_image"):
            self._get_lens_image()
        
        model = self.lens_image.model(
            unconvolved=True,
            kwargs_mass=self.mass_kwargs,
            kwargs_light=self.light_kwargs,
            eta_flat=self.eta
        )

        return model

    def get_simulation(self) -> np.ndarray:
        """
        initialises LensImage and returns an array of the lens model
        """
        if not hasattr(self, "lens_image"):
            self.get_lens_image()
        
        model = self.lens_image.simulation(
            kwargs_mass=self.mass_kwargs,
            kwargs_light=self.light_kwargs,
            eta_flat=self.eta
        )

        return model


    def get_source(self):
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        return self.light_model.surface_brightness(x=x_grid, y=y_grid, kwargs=self.light_kwargs)
    

    def get_magnification(self):
        """
        get the magnification map of the lensing system
        
        Returns
        -------
        mu_wrts1 : array
            magnification considering only the lens mass
        mu_wrts2: array
            magnification considering source1 and lens mass
        """
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        mu_wrtlens, mu_wrts1, mu_wrts2 = 1/self.mass_model.inverse_magnification(x=x_grid, y=y_grid, kwargs=self.mass_kwargs, eta_flat=self.eta)

        return mu_wrts1, mu_wrts2

    def get_inverse_magnification(self):
        """
        get the imverse magnification map of the lensing system
        
        Returns
        -------
        mu_inv_wrts1 : array
            inverse magnification considering only the lens mass
        mu_inv_wrts2: array
            inverse magnification considering source1 and lens mass
        """
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        mu_inv_wrtlens, mu_inv_wrts1, mu_inv_wrts2 = self.mass_model.inverse_magnification(x=x_grid, y=y_grid, kwargs=self.mass_kwargs, eta_flat=self.eta)

        return mu_inv_wrts1, mu_inv_wrts2
    

    def get_critical_curves(self):
        """
        get the critical curves at lens and soure1 plane

        Returns
        -------
        cc_wrts1 : list[points]
            A list of points which give the lens effective critical curves. Each curve is a different nested list.
        contours_phys_s1 : list[points]
            A list of points which give the source1 critical curves. Each curve is a different nested list.
        """
        mu_inv_wrts1, mu_inv_wrts2 = self.get_inverse_magnification()
        mu_inv_wrts1 = np.array(mu_inv_wrts1)
        mu_inv_wrts2 = np.array(mu_inv_wrts2)

        contours_wrts1 = measure.find_contours(mu_inv_wrts1, level=0.0)
        contours_wrts2 = measure.find_contours(mu_inv_wrts2, level=0.0)

        xmin, xmax, ymin, ymax = self.pixel_grid.extent
        ny_s1, nx_s1 = mu_inv_wrts1.shape
        ny_s2, nx_s2 = mu_inv_wrts2.shape

        cc_wrts1 = []
        for c in contours_wrts1:
            x = xmin + (c[:, 1] / (nx_s1 - 1)) * (xmax - xmin)
            y = ymin + (c[:, 0] / (ny_s1 - 1)) * (ymax - ymin)
            cc_wrts1.append(np.column_stack([x, y]))

        cc_wrts2 = []
        for c in contours_wrts2:
            x = xmin + (c[:, 1] / (nx_s2 - 1)) * (xmax - xmin)
            y = ymin + (c[:, 0] / (ny_s2 - 1)) * (ymax - ymin)
            cc_wrts2.append(np.column_stack([x, y]))

        return cc_wrts1, cc_wrts2

    def get_caustics(self):
            """
            get the caustics due to lens and source1 

            Returns
            -------
            contours_phys_lens : list[points]
                A list of points which give the lens effective critical curves. Each curve is a different nested list.
            contours_phys_s1 : list[points]
                A list of points which give the source1 critical curves. Each curve is a different nested list.
            """
            cc_wrts1, cc_wrts2 = self.get_critical_curves()

            s1_caustics = []
            for curve in cc_wrts1:
                x_img = jnp.array(curve[:, 0])
                y_img = jnp.array(curve[:, 1])

                x_def, y_def = self.mass_model.ray_shooting(x_img, y_img, eta_flat=self.eta, kwargs=self.mass_kwargs)

                x_s1 = x_def.T[:, 1]
                y_s1 = y_def.T[:, 1]

                s1_caustics.append(np.column_stack([np.array(x_s1), np.array(y_s1)]))


            s2_caustics = []
            for curve in cc_wrts2:
                x_img = jnp.array(curve[:, 0])
                y_img = jnp.array(curve[:, 1])

                x_def, y_def = self.mass_model.ray_shooting(x_img, y_img, eta_flat=self.eta, kwargs=self.mass_kwargs)

                x_s2 = x_def.T[:, 2]
                y_s2 = y_def.T[:, 2]

                s2_caustics.append(np.column_stack([np.array(x_s2), np.array(y_s2)]))


            return s1_caustics, s2_caustics


    def get_convergence(self):
        """
        get the effective convergence map of lens and source1 planes

        Returns
        -------
        lens_kappa_wrts1 : array
            the effective convergence map of the lens
        lens_kappa_wrts2 : array
            the convergence map of the system
        """
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        lens_kappa_wrtlens, lens_kappa_wrts1, lens_kappa_wrts2 = self.mass_model.kappa(x=x_grid, y=y_grid, kwargs=self.mass_kwargs, eta_flat=self.eta)

        return lens_kappa_wrts1, lens_kappa_wrts2

    def get_shear(self):
        """
        get the shear map of the lens and source1 planes

        Returns
        --------
        gamma_1_wrts1 : array
            the gamma1 map of the lens plane
        gamma_2_wrts1 : array
            the gamma2 map of the lens plane
        gamma_1_wrts2 : array
            the gamma1 map of the system
        gamma_2_wrts2 : array
            the gamma2 map of the system
        """
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        gamma_1, gamma_2 = self.mass_model.gamma(x=x_grid, y=y_grid, kwargs=self.mass_kwargs, eta_flat=self.eta)

        gamma_1_wrtlens, gamma_1_wrts1, gamma_1_wrts2 = gamma_1
        gamma_2_wrtlens, gamma_2_wrts1, gamma_2_wrts2 = gamma_2

        return gamma_1_wrts1, gamma_2_wrts1, gamma_1_wrts2, gamma_2_wrts2
    

    """re-generate all necessary parameters"""

    def shuffle(self):
        self.mass_kwargs, self.light_kwargs, self.eta = self._initialise_profile_parameters()

    def n_models(self, filepath: str, n=10):
        """
        simulate n lens images and save them to filepath
        """
        for i in range(n):
            self.shuffle()
            sim = self.get_model()
            plt.imshow(sim, extent=self.pixel_grid.extent, cmap='PuOr_r', norm='log') # type: ignore
            plt.colorbar()
            plt.title(f"model {i}") 
            plt.savefig(filepath + f"model{i}.png")
            plt.close()

    def n_simulations(self, filepath: str, n=10):
        """
        simulate n lens images and save them to filepath
        """
        for i in range(n):
            self.shuffle()
            sim = self.get_simulation()
            plt.imshow(sim, extent=self.pixel_grid.extent, cmap='PuOr_r') # type: ignore
            plt.colorbar()
            plt.title(f"simulation {i}") 
            plt.savefig(filepath + f"simulation_{i}.png")
            plt.close()

        
        