import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

# for finding 0 contours of magnification
from skimage import measure
from scipy.interpolate import splprep, splev


from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.MassModel.mass_model_base import MassModelBase
from herculens.LightModel.light_model_base import LightModelBase

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image import LensImage

class Generator():

    PIXEL_SCALE = {'hubble': 0.04, 'euclid': 0.1, 'JWST': 0.063, 'example': 0.01} # arcseconds
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
                 source_img: str | np.ndarray | None = None,
                 telescope: str = 'euclid',
                 exposure_time: float = 6e2,
                 lens_amp = 1.0,
                 source_amp = 1.0
                 ) -> None:
        
        self.fov = fov
        
        if isinstance(lens_img, str):
            lens_img = np.array(np.load(lens_img))
        self.lens_img = lens_img

        if isinstance(source_img, str):
            source_img = np.array(np.load(source_img))
        self.source_img = source_img

        self.telescope = telescope

        self.exposure_time = exposure_time
        self.lens_amp = lens_amp
        self.source_amp = source_amp

        self.pixel_grid = self._initialise_pixel_grid()
        self.noise, self.psf = self._initialise_noise_psf()
        self.lens_mass_model, self.lens_light_model, self.source_light_model = self._initialise_models()
        self.lens_mass_kwargs, self.lens_light_kwargs, self.source_light_kwargs = self._initialise_profile_parameters()

    # --- these are the same in MPGenerator --- #
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
    

    def _initialise_models(self) -> tuple[MassModel, LightModel, LightModel]:
        """
        Initialise mass and light models of all planes in system
        
        Returns
        -------
        lens_mass_model : MassModel
            lens mass
        lens_light_model : LightModel
            lens light
        source_light_model: LightModel
            source light
        """
        # TODO create lens image class to make setting the lens image easier

        lens_mass_model = MassModel(['EPL', 'SHEAR'], ) 
        if isinstance(self.lens_img, np.ndarray):
            lens_light_model = self._initialise_lens_light_pixelated()
        else:
            lens_light_model = LightModel(['SERSIC_ELLIPSE'])

        if isinstance(self.source_img, np.ndarray):
            source_light_model = self._initialise_source_light_pixelated()
        else:
            source_light_model = LightModel(['SERSIC_ELLIPSE'])

        return lens_mass_model, lens_light_model, source_light_model

    def _initialise_lens_light_pixelated(self) -> LightModel:
        """
        if self.lens_image is an array, initialises and returns a LightModel(['PIXELTED']) instance 
        """
        assert isinstance(self.lens_img, np.ndarray)
        ny_img, nx_img = self.lens_img.shape 

        px_scale_x = float(self.fov) / float(nx_img)
        px_scale_y = float(self.fov) / float(ny_img)
        px_scale_img = 0.5 * (px_scale_x + px_scale_y)

        position = (0, 0)
        spiral_kwargs = {
            'grid_center': position,
            # keep physical grid extents in arcsec (same FOV as this generator)
            'grid_shape': (self.fov, self.fov),
            'pixel_scale_factor': px_scale_img / self.PIXEL_SCALE[self.telescope]
        }
        lens_light_model = LightModel(['PIXELATED'], pixel_interpol='bilinear', kwargs_pixelated=spiral_kwargs) 
        return lens_light_model
    
    def _initialise_source_light_pixelated(self) -> LightModel:
        """
        if self.sourec_image is an array, initialises and returns a LightModel(['PIXELTED']) instance 
        """
        assert isinstance(self.source_img, np.ndarray)
        ny_img, nx_img = self.source_img.shape 

        px_scale_x = float(self.fov) / float(nx_img)
        px_scale_y = float(self.fov) / float(ny_img)
        px_scale_img = 0.5 * (px_scale_x + px_scale_y)

        position = (0, 0)
        spiral_kwargs = {
            'grid_center': position,
            # keep physical grid extents in arcsec (same FOV as this generator)
            'grid_shape': (self.fov, self.fov),
            'pixel_scale_factor': px_scale_img / self.PIXEL_SCALE[self.telescope]
        }
        lens_light_model = LightModel(['PIXELATED'], pixel_interpol='bilinear', kwargs_pixelated=spiral_kwargs) 
        return lens_light_model

    # --- all random generation happens in here --- #
    def _initialise_profile_parameters(self) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
        """
        initialise mass and light profile parameters for lens and source

        Returns
        -------
        lens_mass_kwargs : list[dict[str, float]]
        lens_light_kwargs : list[dict[str, float]]
        source_light_kwargs : list[dict[str, float]]
        """ 

        # -- lens mass -- #
        def compute_ellipticity(q, phi):
            e = (1 - q) / (1 + q)
            e1 = e * np.cos(2 * phi)
            e2 = e * np.sin(2 * phi)
            return e1, e2
        
        e1, e2 = compute_ellipticity(
            q = np.random.uniform(0.4, 1.0), 
            phi = np.random.uniform(0, np.pi)
        )

        lens_EPL_kwargs = {
            'theta_E': np.random.uniform(0.5, 1.0),
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
        lens_mass_kwargs = [lens_EPL_kwargs, lens_shear_kwargs]

        # -- lens light -- #
        if "PIXELATED" in self.lens_light_model.profile_type_list:
            assert isinstance(self.lens_img, np.ndarray)
            img = self.lens_amp * self.lens_img / np.max(self.lens_img)
            lens_light_kwargs = [{
                "pixels" : img
            }]
        elif "SERSIC_ELLIPSE" in self.lens_light_model.profile_type_list:
            lens_light_kwargs = [{
                'amp': self.lens_amp,
                'R_sersic': lens_EPL_kwargs['theta_E']*1e-2,
                'n_sersic': 4.0, # n=4: de Vaucouleurs profile
                'e1': lens_EPL_kwargs['e1'],
                'e2': lens_EPL_kwargs['e2'],
                'center_x': 0.0,
                'center_y': 0.0
            }]
        else: # NOTE for debugging, can remove later
            raise ValueError(f"something has gone wrong here")

        # -- source light -- #
        if "PIXELATED" in self.source_light_model.profile_type_list:
            assert isinstance(self.source_img, np.ndarray)
            img = self.source_amp * self.source_img / np.max(self.source_img)
            source_light_kwargs = [{
                "pixels" : img
            }]
        elif "SERSIC_ELLIPSE" in self.source_light_model.profile_type_list:
            source_light_kwargs = [{
                'amp': self.lens_amp,
                'R_sersic': lens_EPL_kwargs['theta_E']*1e-3,
                'n_sersic': 4.0, # n=4: de Vaucouleurs profile
                'e1': 0.0,
                'e2': 0.1,
                'center_x': np.random.normal(0.0, lens_EPL_kwargs['theta_E']),
                'center_y': np.random.normal(0.0, lens_EPL_kwargs['theta_E'])
            }]
        else: # NOTE for debugging, can remove later
            raise ValueError(f"something has gone wrong here")

        return lens_mass_kwargs, lens_light_kwargs, source_light_kwargs
    
    # ---  --- #

    """getter methods"""

    def get_lens_image(self) -> tuple[LensImage, list[dict[str,float]], list[dict[str,float]], list[dict[str,float]]]:
        """
        get the MPLensImage instance of this shuffle

        Returns
        -------
        lens_image : LensImage
            the lens image instance
        lens_mass_kwargs: list[dict[str,float]]
            A nested list of dictionaries which give the lens mass profile
        lens_light_kwargs: list[dict[str,float]]
            A nested list of dictionaries which give the lens light profile
        source_light_kwargs : list[dict[str,float]]
            A nested list of dictionaries which give the source light profile
        """

        lens_image = LensImage(
            grid_class=self.pixel_grid,
            psf_class=self.psf,
            noise_class=self.noise,
            lens_mass_model_class=self.lens_mass_model,
            lens_light_model_class=self.lens_light_model,
            source_model_class=self.source_light_model
        )
        self.lens_image = lens_image
        return lens_image, self.lens_mass_kwargs, self.lens_light_kwargs, self.source_light_kwargs
            
    def get_model(self) -> np.ndarray:
        """
        initialises LensImage and returns an array of the lens model
        """
        if not hasattr(self, "lens_image"):
            self.get_lens_image()
        
        model = self.lens_image.model(
            unconvolved=True,
            kwargs_lens=self.lens_mass_kwargs,
            kwargs_lens_light=self.lens_light_kwargs,
            kwargs_source=self.source_light_kwargs
        )

        return model
    
    def get_simulation(self) -> np.ndarray:
        """
        initialises LensImage and returns an array of the lens simulation
        """
        if not hasattr(self, "lens_image"):
            self.get_lens_image()
        
        model = self.lens_image.simulation(
            kwargs_lens=self.lens_mass_kwargs,
            kwargs_lens_light=self.lens_light_kwargs,
            kwargs_source=self.source_light_kwargs,
            prng_key=random.PRNGKey(np.random.randint(0, 10)) # NOTE can make this different
        )

        return model
        
    def get_source(self):
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        return self.source_light_model.surface_brightness(x=x_grid, y=y_grid, kwargs=self.source_light_kwargs)
    
    def get_magnification(self):
        """returns the magnification map of the lensing system"""
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        mu = 1/self.lens_mass_model.inverse_magnification(x=x_grid, y=y_grid, kwargs=self.lens_mass_kwargs)

        return mu
    
    def get_inverse_magnification(self):
        """returns the magnification map of the lensing system"""
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        mu_inv = self.lens_mass_model.inverse_magnification(x=x_grid, y=y_grid, kwargs=self.lens_mass_kwargs)

        return mu_inv

    def get_convergence(self):
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid

        kappa = self.lens_mass_model.kappa(x=x_grid, y=y_grid, kwargs=self.lens_mass_kwargs)

        return kappa

    def get_shear(self):
        x_grid = self.pixel_grid._x_grid
        y_grid = self.pixel_grid._y_grid
        gamma_1, gamma_2 = self.lens_mass_model.gamma(x=x_grid, y=y_grid, kwargs=self.lens_mass_kwargs)

        return gamma_1, gamma_2

    def get_critical_curves(self):
        mu_inv = np.asarray(self.get_inverse_magnification()).copy()

        contours_pix = measure.find_contours(mu_inv, level=0.0)

        xmin, xmax, ymin, ymax = self.pixel_grid.extent
        ny, nx = mu_inv.shape

        contours_phys = []
        for c in contours_pix:
            x = xmin + (c[:, 1] / (nx - 1)) * (xmax - xmin)
            y = ymin + (c[:, 0] / (ny - 1)) * (ymax - ymin)
            contours_phys.append(np.column_stack([x, y]))

        return contours_phys

    def get_caustics(self):
        critical_curves = self.get_critical_curves()

        caustics = []
        for curve in critical_curves:
            x_img = jnp.array(curve[:, 0])
            y_img = jnp.array(curve[:, 1])

            x_src, y_src = self.lens_mass_model.ray_shooting(x_img, y_img, self.lens_mass_kwargs)

            caustics.append(np.column_stack([np.array(x_src), np.array(y_src)]))

        return caustics
        
        

