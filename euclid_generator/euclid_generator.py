import numpy as np

import jax
import jax.numpy as jnp
from jax_zero_contour import ZeroSolver
from jax import random

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.path import Path

from skimage import measure
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay

from astropy.io import fits

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.Util import param_util
from herculens.Util.param_util import phi_q2_ellipticity

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage

from correlated_noise import K_grid, P_Matern, pack_fft_values

import os
from tqdm import tqdm
import copy


class EuclidGenerator:

    PIXEL_SCALE = 0.1       # arseconds/pixel
    BACKGROUND_RMS = 4.2    # e- rms
    PSF_FWHM = 0.15         # arcseconds

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
        self.n_sims = 4*len(self.lens_files)

        self.pixel_grid = self.__initialise_pixel_grid()
        self.noise = self.__initialise_noise()
        self.psf = self.__initialise_psf()

        self.mass_model, self.light_model = self.__initialise_models()
        self.LensImage = self.__initialise_LensImage()

        # TODO should these be class attributes or just variables that get passed around?

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

    # for initialising source light classes
    # NOTE supersampled with factor of two. This may need changing if it is too slow =(
    SOURCE_PIXEL_COORDS = np.meshgrid(np.linspace(-7.5, 7.5, 550), np.linspace(-7.5, 7.5, 550))

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
            img = hdul[1].data # type: ignore
        lens_light_model = self.__initialise_light_pixelated(img)

        # source1_light_model = LightModel(['SERSIC_ELLIPSE'])
        source1_light_model = self.__initialise_light_pixelated(EuclidGenerator.SOURCE_PIXEL_COORDS[0])

        # source2_light_model = LightModel(['SERSIC_ELLIPSE'])
        source2_light_model = self.__initialise_light_pixelated(EuclidGenerator.SOURCE_PIXEL_COORDS[0])

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
    def __get_shape_of_source(source_img, threshold=0.1):
        """
        Calculates the axis ratio and position angle from the image of a source, returns as (e1, e2) ellipticity parameters.

        Method from Dan!!
        """

        mask = source_img > (threshold * source_img.max()) # remove noise I think

        y, x = np.indices(source_img.shape)
        # x, y = source_coords
        I = source_img * mask  # masked image

        # Compute centroid
        x_c = (I * x).sum() / I.sum()
        y_c = (I * y).sum() / I.sum()

        # Centered coordinates
        x_cen = x - x_c
        y_cen = y - y_c

        # Second moments
        Mxx = (I * x_cen**2).sum() / I.sum()
        Myy = (I * y_cen**2).sum() / I.sum()
        Mxy = (I * x_cen * y_cen).sum() / I.sum()

        # Inertia tensor
        M = jnp.array([[Mxx, Mxy],
                    [Mxy, Myy]])

        # Eigen decomposition
        evals, evecs = jnp.linalg.eigh(M)
        order = jnp.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        major, minor = jnp.sqrt(evals)
        axis_ratio = minor / major
        pos_angle = jnp.arctan2(evecs[1, 0], evecs[0, 0])  # radians
        
        e1_src, e2_src = param_util.phi_q2_ellipticity(pos_angle, axis_ratio)

        return e1_src, e2_src

    @staticmethod
    def __sample_from_pdf(pdf_filepath, key):
        data = jnp.load(pdf_filepath)
        x, pdf = data[:, 0], data[:, 1]

        dx = jnp.diff(x, append=x[-1])  
        cdf = jnp.cumsum(pdf * dx)
        cdf /= cdf[-1]  

        val = random.uniform(key, minval=0.0, maxval=1.0)
        samples = jnp.interp(val, cdf, x)

        return samples
    
    @staticmethod
    def __sample_point_in_polygon_triangulation(vertices, key):
        # Triangulate polygon
        tri = Delaunay(vertices)

        triangles = vertices[tri.simplices]  # shape (n_tri, 3, 2)

        # Compute triangle areas
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]

        areas = 0.5 * np.abs(
            (v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) -
            (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1])
        )

        # Choose triangle weighted by area
        probs = areas / areas.sum()
        idx = np.random.choice(len(triangles), p=probs)

        triangle = triangles[idx]

        # Sample uniformly inside triangle
        keys = random.split(key, 2)
        r1 = random.uniform(keys[0], minval=0.0, maxval=1.0)
        r2 = random.uniform(keys[1], minval=0.0, maxval=1.0)

        sqrt_r1 = np.sqrt(r1)

        point = (
            (1 - sqrt_r1) * triangle[0] +
            sqrt_r1 * (1 - r2) * triangle[1] +
            sqrt_r1 * r2 * triangle[2]
        )

        return point

    @staticmethod
    def __sample_point_in_polygon_rejection_sampling(vertices, key):
    
        points = np.array(vertices)
        path = Path(points)
        min_x, min_y = points[:,0].min(), points[:,1].min()
        max_x, max_y = points[:,0].max(), points[:,1].max()
        
        i = 0
        keys = random.split(key, 100)
        while True:
            px = float(random.uniform(key=keys[i], minval=min_x, maxval=max_x))
            py = float(random.uniform(key=keys[i+49], minval=min_y, maxval=max_y))
            if path.contains_point((px, py)):
                return px, py
            i += 1


    # for generating a fake system to compute caustics # 
    FAUX_S1_KWARGS = [{
        'theta_E' : 0.0, 
        'center_x': 0.0,    
        'center_y': 0.0    
    },{
        'gamma1': 0.0,     
        'gamma2': 0.0,      
        'ra_0': 0.0,
        'dec_0': 0.0,
    }]

    FAUX_LENS_IMG = jnp.zeros((150,150))

    def __sample_lens_kwargs(self, key, image_index, lens_rotation) -> tuple[jnp.ndarray, list[dict[str, float]], list[dict[str, float]]]:
        """
        Samples kwargs for the lens model based on the image

        Returns
        -------
        lens_light_kwargs : list[dict[str : float]]
        
        lens_mass_kwargs : list[dict[str : float]]
        """
        keys = random.split(key, 4)

        with fits.open(self.lens_folder_path + self.lens_files[image_index]) as hdul:
            img = jnp.asarray(hdul[1].data, dtype=jnp.float32) # type: ignore
            rms = np.asarray(hdul[3].data, dtype=jnp.float32) # type: ignore

        if lens_rotation in [1, 2, 3]:
            img = jnp.rot90(img, k=lens_rotation)
            rms = np.rot90(rms, k=lens_rotation)

        lens_light_kwargs = [{
            "pixels" : EuclidGenerator.FAUX_LENS_IMG
        }]
        
        # Take the center of the image and get shape of source ( assumes that center is at (0,0) )
        e1_lens, e2_lens = EuclidGenerator.__get_shape_of_source(img[70:90,70:90])
        lens_EPL_kwargs = {
            'theta_E' : EuclidGenerator.__sample_from_pdf(EuclidGenerator.THETA_PDF_FILEPATH, keys[0]), 
            'gamma': random.uniform(key=keys[1], minval=1.7, maxval=2.3),
            'e1': e1_lens,
            'e2': e2_lens,
            'center_x': 0.0,
            'center_y': 0.0
        }
        lens_shear_kwargs = {
            'gamma1': random.uniform(key=keys[2], minval=-0.3, maxval=0.3),
            'gamma2': jnp.float32(random.uniform(key=keys[3], minval=-0.3, maxval=0.3)),
            'ra_0': lens_EPL_kwargs['center_x'],
            'dec_0': lens_EPL_kwargs['center_y'],
        }
        lens_mass_kwargs = [lens_EPL_kwargs, lens_shear_kwargs]

        return self.lens_amp*img, lens_mass_kwargs, lens_light_kwargs
    
    def __sample_s1_kwargs(self, key, tangential_caustic):
        """
        Samples kwargs for the lens model based on the image

        Returns
        -------
        lens_light_kwargs : list[dict[str : float]]
        
        lens_mass_kwargs : list[dict[str : float]]
        """
        keys = random.split(key, 10)

        # method 1: #
        # downscaling_factor = 2
        # xmin = jnp.min(tangential_caustic[:, 0])/downscaling_factor
        # xmax = jnp.max(tangential_caustic[:, 0])/downscaling_factor
        # ymin = jnp.min(tangential_caustic[:, 1])/downscaling_factor
        # ymax = jnp.max(tangential_caustic[:, 1])/downscaling_factor

        # method 2: #
        # cx_s1, cy_s1 = EuclidGenerator.__sample_point_in_polygon_triangulation(tangential_caustic, keys[3])
 
        # method 3: #
        cx_s1, cy_s1 = EuclidGenerator.__sample_point_in_polygon_rejection_sampling(tangential_caustic, keys[0])
        
        source1_SIS_kwargs = {
            'theta_E' : EuclidGenerator.__sample_from_pdf(EuclidGenerator.THETA_PDF_FILEPATH, keys[1]), 
            'center_x': cx_s1,    
            'center_y': cy_s1  
        }
        source1_shear_kwargs = {
            'gamma1': random.uniform(key=keys[2], minval=0.0, maxval=0.3),     
            'gamma2': random.uniform(key=keys[3], minval=0.0, maxval=0.3),         
            'ra_0': source1_SIS_kwargs['center_x'],
            'dec_0': source1_SIS_kwargs['center_y'],
        }
        source1_mass_kwargs = [source1_SIS_kwargs, source1_shear_kwargs]

        e1_s1, e2_s1 = phi_q2_ellipticity(
            q = random.uniform(keys[4], minval=0.7, maxval=1.0),  # TODO
            phi = random.uniform(keys[5], minval=0.0, maxval=2*jnp.pi)
        )
        # how am I going to add noise to this?

        source1_SERSIC_kwargs = {
            'amp': self.source1_amp,
            'R_sersic': random.uniform(key=keys[6], minval=0.1, maxval=1.0),     
            'n_sersic' : random.uniform(key=keys[7], minval=0.5, maxval=4.0),
            'e1': e1_s1,
            'e2': e2_s1,
            'center_x': source1_SIS_kwargs['center_x'], 
            'center_y': source1_SIS_kwargs['center_y']
        }

        sersic_light = LightModel(['SERSIC_ELLIPSE']).surface_brightness(*EuclidGenerator.SOURCE_PIXEL_COORDS, kwargs=[source1_SERSIC_kwargs])

        n = 1
        sigma = 0.3
        rho = 20

        k_grid = K_grid(shape=sersic_light.shape, scale=1)
        P = P_Matern(k_grid.k, n, sigma, rho, k_zero=0)

        scale = jnp.sqrt(P)
        white_noise = random.normal(key=keys[8], shape=sersic_light.shape)
        
        pixels = jnp.fft.irfft2(pack_fft_values(white_noise*scale), s=scale.shape, norm="ortho")
        pixels -= jnp.mean(pixels)
        pixels /= jnp.std(pixels)

        alpha = 0.6  # strength of perturbation
        pixels = 1 + alpha * pixels

        pixels = jnp.log1p(jnp.exp(alpha * pixels))
        sersic_light = sersic_light * pixels

        source1_light_kwargs = [{
            'pixels' : sersic_light
        }]

        return source1_mass_kwargs, source1_light_kwargs
    
    def __sample_s2_kwargs(self, key, tangential_caustic):

        keys = random.split(key, 10)

        e1_s2, e2_s2 = phi_q2_ellipticity(
            q = random.uniform(keys[0], minval=0.7, maxval=1.0),  # TODO
            phi = random.uniform(keys[1], minval=-jnp.pi, maxval=jnp.pi)
        )

        cx_s2, cy_s2 = EuclidGenerator.__sample_point_in_polygon_rejection_sampling(tangential_caustic, keys[2])
        source2_SERSIC_kwargs = {
            'amp': self.source2_amp,
            'R_sersic': random.uniform(key=keys[3], minval=0.1, maxval=0.5),     
            'n_sersic' : random.uniform(key=keys[4], minval=0.5, maxval=1.5),
            'e1': e1_s2,         
            'e2': e2_s2,         
            'center_x': cx_s2, 
            'center_y': cy_s2
        }
        sersic_light = LightModel(['SERSIC_ELLIPSE']).surface_brightness(*EuclidGenerator.SOURCE_PIXEL_COORDS, kwargs=[source2_SERSIC_kwargs])

        n = 1
        sigma = 0.3
        rho = 20

        k_grid = K_grid(shape=sersic_light.shape, scale=1)
        P = P_Matern(k_grid.k, n, sigma, rho, k_zero=0)

        scale = jnp.sqrt(P)
        white_noise = random.normal(key=keys[5], shape=sersic_light.shape)
        
        pixels = jnp.fft.irfft2(pack_fft_values(white_noise*scale), s=scale.shape, norm="ortho")
        pixels -= jnp.mean(pixels)
        pixels /= jnp.std(pixels)

        alpha = 0.6  # strength of perturbation
        pixels = 1 + alpha * pixels

        pixels = jnp.log1p(jnp.exp(alpha * pixels))
        sersic_light = sersic_light * pixels

        source2_light_kwargs = [{
            'pixels' : sersic_light
        }]

        return source2_light_kwargs

    def __sample_kwargs(self, key, image_index, lens_rotation=0):
        """
        sample all kwargs for next random generation of lenses
        """

        keys = random.split(key, 5)
        eta = jnp.float32(self.__sample_from_pdf(EuclidGenerator.ETA_PDF_FILEPATH, keys[0]))
        
        # --- lens --- #
        lens_img, lens_mass_kwargs, lens_light_kwargs = self.__sample_lens_kwargs(keys[1], image_index, lens_rotation)
        
        faux_mp_mass_kwargs = [lens_mass_kwargs, EuclidGenerator.FAUX_S1_KWARGS]

        # -- compute bounds on s1, s2 position with tangenial caustic -- #
        tangential_caustic_s1 = np.array(self.get_tangential_caustic(faux_mp_mass_kwargs, eta, 1))
        source1_mass_kwargs, source1_light_kwargs = self.__sample_s1_kwargs(keys[2], tangential_caustic_s1)

        mp_mass_kwargs = [lens_mass_kwargs, source1_mass_kwargs]

        # --- source 2 --- #
        tangential_caustic_s2 = np.array(self.get_tangential_caustic(mp_mass_kwargs, eta, 2))
        source2_light_kwargs = self.__sample_s2_kwargs(keys[3], tangential_caustic_s2) 

        mp_light_kwargs = [lens_light_kwargs, source1_light_kwargs, source2_light_kwargs]

        return lens_img, mp_mass_kwargs, mp_light_kwargs, eta


    def get_inverse_magnification(self, x, y, mass_kwargs, eta, plane):
        A = self.mass_model.A( x=x, y=y, kwargs=mass_kwargs, eta_flat=eta )
        Ap = A[plane, :, :]
        return Ap[..., 0, 0] * Ap[..., 1, 1] - Ap[..., 0, 1] * Ap[..., 1, 0]

    def get_critical_curves(self, mass_kwargs, eta, plane=1):

        x_vec = jnp.linspace(-20.0, 20.0, 1000)
        y_vec = jnp.linspace(-20.0, 20.0, 1000)
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
            x_img = jnp.array(curve)[0]
            y_img = jnp.array(curve)[1]

            x_def, y_def = self.mass_model.ray_shooting(x_img, y_img, eta_flat=eta, kwargs=mass_kwargs)

            x_s1 = x_def.T[:, plane]
            y_s1 = y_def.T[:, plane]

            caustics.append(np.column_stack([np.array(x_s1), np.array(y_s1)]))

        return caustics
    
    def get_tangential_caustic(self, mass_kwargs, eta, plane=1):
        caustics = self.get_caustics(mass_kwargs, eta, plane)

        if not caustics:
            return None

        return max(caustics, key=lambda curve: len(curve[0]))

    def get_model(self, mass_kwargs, light_kwargs, eta, unconvolved=False, source=None) -> np.ndarray:

        if source is None:
            model = self.LensImage.model(
                kwargs_mass=mass_kwargs,
                kwargs_light=light_kwargs,
                eta_flat=eta,
                unconvolved=unconvolved
            )

        elif source == 1:
            light_kwargs_ = copy.deepcopy(light_kwargs)
            light_kwargs_[2][0]['pixels'] *= 0.0   # zero source 1

            model = self.LensImage.model(
                kwargs_mass=mass_kwargs,
                kwargs_light=light_kwargs_,
                eta_flat=eta,
                unconvolved=unconvolved
            )

        elif source == 2:
            light_kwargs_ = copy.deepcopy(light_kwargs)
            light_kwargs_[1][0]['pixels'] *= 0.0   # zero source 2

            model = self.LensImage.model(
                kwargs_mass=mass_kwargs,
                kwargs_light=light_kwargs_,
                eta_flat=eta,
                unconvolved=unconvolved
            )

        return model

    def get_simulation(self, mass_kwargs, light_kwargs, eta, noise_key) -> np.ndarray:
        
        sim = self.LensImage.simulation(
            kwargs_mass=mass_kwargs,
            kwargs_light=light_kwargs,
            eta_flat=eta,
            noise_seed=noise_key
        )

        return sim

    @staticmethod
    def generate_preso_figures1(key=0, i=2000):
        s1_amp = 160
        s2_amp = 160
        gen = EuclidGenerator("euclid_generator/lrg_used_for_karina_sim/", 15, lens_amp=1, source1_amp=s1_amp, source2_amp=s2_amp)
        # 6
        mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(random.PRNGKey(key), i) # 22, 1000, 2000 are good ones

        x_grid, y_grid = gen.pixel_grid.pixel_coordinates
        lens_img, s1_img, s2_img = gen.light_model.surface_brightness(
            [x_grid, x_grid, x_grid], 
            [y_grid, y_grid, y_grid], 
            light_kwargs,
            [x_grid, x_grid, x_grid], 
            [y_grid, y_grid, y_grid]
        )

        # --- lens images --- #
        
        plt.imshow(lens_img, norm='symlog', extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("lens image log scale")
        plt.savefig("euclid_generator/presentation_figures/lens_log.png")

        plt.close()

        plt.imshow(lens_img, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("LRG lens image")
        plt.savefig("euclid_generator/presentation_figures/lens.png")

        plt.close()

        # --- source images --- #
        x_vec = np.linspace(-15, 15, 300)
        y_vec = np.linspace(-15, 15, 300)
        X, Y = np.meshgrid(x_vec, y_vec)
        mu_1 = gen.get_inverse_magnification(X, Y, mass_kwargs, eta, plane=1)
        mu_2 = gen.get_inverse_magnification(X, Y, mass_kwargs, eta, plane=2)

        plt.imshow(mu_1, extent=gen.pixel_grid.extent, cmap="PuOr_r", norm="symlog")
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/magnification_s1.png")
        plt.close()

        plt.imshow(mu_2, extent=gen.pixel_grid.extent, cmap="PuOr_r", norm="symlog")
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/magnification_s2.png")
        plt.close()

        tang = np.array(gen.get_tangential_caustic(mass_kwargs, eta, 1))
        tang2 = np.array(gen.get_tangential_caustic(mass_kwargs, eta, 2))

        plt.plot(*tang.T, label="tangential caustic s1")
        plt.plot(*tang2.T, label="tangential caustic s2")
        plt.axis('equal')
        plt.legend()
        plt.savefig("euclid_generator/presentation_figures/caustics_empty.png")
        plt.close()

        s1_cx = light_kwargs[1][0]['center_x']
        s1_cy = light_kwargs[1][0]['center_y']
        s2_cx = light_kwargs[2][0]['center_x']
        s2_cy = light_kwargs[2][0]['center_y']

        plt.plot(*tang.T, label="tangential caustic s1")
        plt.plot(*tang2.T, label="tangential caustic s2")
        plt.scatter(s1_cx, s1_cy, label="source 1 position", marker='x')
        plt.scatter(s2_cx, s2_cy, label="source 2 position", marker='x')
        plt.legend()
        plt.title("source positions")
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/caustics_w_source_positions.png")

        plt.close()
        

        plt.plot(*tang2.T, label="tangenntial caustic")
        plt.scatter(s2_cx, s2_cy, label="source 2 position", marker='x')
        plt.legend()
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/caustics_w_s2_position.png")

        plt.close()

        plt.plot(*tang.T, label="tangenntial caustic")
        plt.scatter(s1_cx, s1_cy, label="source 1 position", marker='x')
        plt.legend()
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/caustics_w_s1_position.png")

        plt.close()

        _, s1_img, s2_img = gen.light_model.surface_brightness(
            [x_grid]*3, 
            [y_grid]*3, 
            light_kwargs,
            [x_grid]*3, 
            [y_grid]*3
        )

        plt.imshow(s1_img, norm="log", extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("source 1 light profile (log scale)")
        plt.savefig("euclid_generator/presentation_figures/s1_log.png")

        plt.close()

        plt.imshow(s2_img, norm="log", extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("source 2 light profile (log scale)")
        plt.savefig("euclid_generator/presentation_figures/s2_log.png")

        plt.close()

        plt.imshow(s1_img, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("source 1 light profile")
        plt.savefig("euclid_generator/presentation_figures/s1.png")

        plt.close()

        plt.imshow(s2_img, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("source 2 light profile")
        plt.savefig("euclid_generator/presentation_figures/s2.png")

        plt.close()

        light_kwargs[0] = [{
            "pixels" : 0 * light_kwargs[0][0]['pixels']
        }]

        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=False)
        plt.imshow(model, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled sources")
        plt.savefig("euclid_generator/presentation_figures/model_no_lens.png")
        plt.close()
        
        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=True)
        plt.imshow(model, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled sources unconvolved")
        plt.savefig("euclid_generator/presentation_figures/model_no_lens_unconvolved.png")
        plt.close()


        light_kwargs[2][0]["amp"] = 0.0

        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=False)
        plt.imshow(model, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled source 1")
        plt.savefig("euclid_generator/presentation_figures/model_no_lens_s1.png")
        plt.close()

        light_kwargs[2][0]["amp"] = s2_amp
        light_kwargs[1][0]["amp"] = 0.0

        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=False)
        plt.imshow(model, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled source 2")
        plt.savefig("euclid_generator/presentation_figures/model_no_lens_s2.png")
        plt.close()

        light_kwargs[1][0]["amp"] = s1_amp
        light_kwargs[2][0]["amp"] = s2_amp
        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=False)
        plt.imshow(model + lens_img, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled DSPL")
        plt.savefig("euclid_generator/presentation_figures/everything.png")

        plt.close()
        
        plt.imshow(model + lens_img, norm="log", extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.title("modelled DSPL (log scale)")
        plt.savefig("euclid_generator/presentation_figures/everything_log.png")

    @staticmethod
    def generate_preso_figures2():
        gen = EuclidGenerator(
            lens_folder_path="euclid_generator/lrg_used_for_karina_sim/",
            fov=15,
            lens_amp=4.0,
            source1_amp=150.0,
            source2_amp=120.0
        )

        x_grid, y_grid = gen.pixel_grid.pixel_coordinates
        lens_img, mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(key=random.PRNGKey(16382), image_index=2000, lens_rotation=3)
        
        print(f"number of images: {gen.n_sims/4}")
        print(f"number of sims: {gen.n_sims}")


        # 1. take real image of RLG
        plt.imshow(lens_img, extent=gen.pixel_grid.extent, norm="symlog") # type: ignore
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/foreground_img.png")
        plt.close()

        # 2. fit ellipticity
        lens_EPL_kwargs = [mass_kwargs[0][0]]
        lens_EPL_model = MassModel(["EPL"])
        kappa = lens_EPL_model.kappa(x_grid, y_grid, lens_EPL_kwargs)

        plt.imshow(lens_img, extent=gen.pixel_grid.extent, norm="symlog") # type: ignore
        plt.colorbar()
        plt.contour(np.log(kappa), extent=gen.pixel_grid.extent, colors="white") # type: ignore
        plt.savefig("euclid_generator/presentation_figures/foreground_img_with_model_fit.png")
        plt.close()

        # 3. sample mass profile (do I need these?)
        lens_mass_kwargs = mass_kwargs[0]
        lens_mass_model = MassModel(["EPL", "SHEAR"])
        kappa = lens_mass_model.kappa(x_grid, y_grid, lens_mass_kwargs)

        plt.imshow(kappa, extent=gen.pixel_grid.extent, norm="log") # type: ignore
        plt.colorbar()
        plt.contour(np.log(kappa), extent=gen.pixel_grid.extent, colors="white") # type: ignore
        plt.savefig("euclid_generator/presentation_figures/lens_kappa.png")
        plt.close()

        # 4. Compute tangential caustics on s1 plane
        tan_caustic_s1 = np.array(gen.get_tangential_caustic(mass_kwargs=mass_kwargs, eta=eta, plane=1))

        plt.plot(*tan_caustic_s1.T, label="s1 tangential caustic", c="orange")
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/s1_tangential_caustic.png")
        plt.close()

        # 5. Randomly place s1 in caustic
        s1_cx = mass_kwargs[1][0]['center_x']
        s1_cy = mass_kwargs[1][0]['center_y']
        
        plt.plot(*tan_caustic_s1.T, label="s1 tangential caustic", c="orange")
        plt.scatter(s1_cx, s1_cy, marker="X", label="s1 position", c="orange")
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/s1_tangential_caustic_with_s1_position.png")
        plt.close()

        # 6. Sample s1 mass & light profiles
        s1_light_kwargs = light_kwargs[1]
        s1_light_model = LightModel(["SERSIC_ELLIPSE"])
        s1_surface_brightness = s1_light_model.surface_brightness(x_grid, y_grid, s1_light_kwargs)
        
        s1_mass_kwargs = mass_kwargs[1]
        s1_mass_model = gen.mass_model.mass_models[1]
        s1_kappa = s1_mass_model.kappa(x_grid, y_grid, s1_mass_kwargs)

        plt.imshow(s1_surface_brightness, extent=gen.pixel_grid.extent, norm="log") # type: ignore
        plt.plot(*tan_caustic_s1.T, label="s1 tangential caustic", c="orange")
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/s1_light_model.png")
        plt.close()

        plt.imshow(s1_surface_brightness, extent=gen.pixel_grid.extent, norm="log") # type: ignore
        plt.colorbar()
        plt.plot(*tan_caustic_s1.T, label="s1 tangential caustic", c="orange")
        plt.contour(np.log(s1_kappa), extent=gen.pixel_grid.extent, colors="white", levels=3) # type: ignore
        plt.savefig("euclid_generator/presentation_figures/s1_mass_light_models.png")
        plt.close()

        # 7. compute s2 caustics
        tan_caustic_s2 = np.array(gen.get_tangential_caustic(mass_kwargs=mass_kwargs, eta=eta, plane=2))

        plt.plot(*tan_caustic_s2.T, label="s2 tangential caustic", c="orange")
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/s2_tangential_caustic.png")
        plt.close()

        s2_cx = light_kwargs[2][0]['center_x']
        s2_cy = light_kwargs[2][0]['center_y']
        
        plt.plot(*tan_caustic_s2.T, label="s2 tangential caustic", c="orange")
        plt.scatter(s2_cx, s2_cy, marker="X", label="s2 position", c="orange")
        plt.axis('equal')
        plt.savefig("euclid_generator/presentation_figures/s2_tangential_caustic_with_s2_position.png")
        plt.close()

        # 8. generate s2 light
        s2_light_model = gen.light_model.light_models[2]
        s2_surface_brightness = s2_light_model.surface_brightness(x_grid, y_grid, light_kwargs[2]) # type: ignore
        
        plt.imshow(s2_surface_brightness, extent=gen.pixel_grid.extent, norm="log") # type: ignore
        plt.colorbar()
        plt.plot(*tan_caustic_s2.T, label="s2 tangential caustic", c="orange")
        plt.savefig("euclid_generator/presentation_figures/s2_light_model.png")
        plt.close()

        # 9. model lens with no lens light (get archs)
        light_kwargs[0] = [{
            "pixels" : 0 * light_kwargs[0][0]['pixels']
        }]

        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=True)
        plt.imshow(model, extent=gen.pixel_grid.extent, norm="symlog")
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/model_no_lens.png")
        plt.close()

        # 10. convolve, paint on top of lens
        model = gen.get_model(mass_kwargs, light_kwargs, eta, unconvolved=False)
        final = model + lens_img

        plt.imshow(final, extent=gen.pixel_grid.extent, norm="symlog")
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/model_w_lens.png")
        plt.close()

        plt.imshow(final, extent=gen.pixel_grid.extent)
        plt.colorbar()
        plt.savefig("euclid_generator/presentation_figures/model_w_lens_notlog.png")
        plt.close()

    @staticmethod
    def big_grid():
        """this is the method I will use to generate final simulations"""
        
        lens_folder_path = "euclid_generator/lrg_used_for_karina_sim/"
        n_files = 6

        gen = EuclidGenerator(
            lens_folder_path=lens_folder_path,
            fov=15,
            lens_amp=80.0,
            source1_amp=150.0,
            source2_amp=120.0
        )

        extent = gen.pixel_grid.extent

        fig, axes = plt.subplots(n_files, 4, figsize=(4*4, 4*n_files))

        # Ensure axes is always 2D
        if n_files == 1:
            axes = np.expand_dims(axes, axis=0)

        start=30
        for i in tqdm(range(n_files)):
            for r in range(4):

                lens_img, mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(
                    random.PRNGKey((i+start)*4 + r),
                    i+start,
                    lens_rotation=r
                )

                model = gen.get_model(mass_kwargs, light_kwargs, eta)

                ax = axes[i, r]
                im = ax.imshow(lens_img + model, extent=extent, norm="symlog")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig("euclid_generator/testing_images/full_grid.png")
        plt.close()

    @staticmethod
    def get_final_sim():
        """this is the method I will use to generate final simulations"""
        
        lens_folder_path = "euclid_generator/lrg_used_for_karina_sim/"
        with os.scandir(lens_folder_path) as entries:
            lens_files = np.array([item.name for item in entries])
        
        # n_files = len(lens_files)
        n_files = 1

        gen = EuclidGenerator(
            lens_folder_path=lens_folder_path,
            fov=15,
            lens_amp=80.0,
            source1_amp=150.0,
            source2_amp=120.0
        )

        extent = gen.pixel_grid.extent
        for i in tqdm(range(n_files)):
            for r in range(0, 4):
                lens_img, mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(random.PRNGKey(i*4+r), i, lens_rotation=r)
                model = gen.get_model(mass_kwargs, light_kwargs, eta)
                
                plt.imshow(lens_img + model, extent=extent, norm="symlog")
                plt.colorbar()
                plt.savefig(f"euclid_generator/testing_images/model({i},{r}).png")
                plt.close()
            
                # fits.writeto(f"euclid_generator/testing_fits/model{i}_{r}.fits", np.array(model))

    @staticmethod
    def testing1():
        lens_folder_path = "euclid_generator/lrg_used_for_karina_sim/"
        
        n_files = 100
        gen = EuclidGenerator(
            lens_folder_path=lens_folder_path,
            fov=15,
            lens_amp=800.0,
            source1_amp=10.0,
            source2_amp=10.0
        )

        extent = gen.pixel_grid.extent
        for i in tqdm(range(n_files)):
            for r in range(0, 4):
                lens_img, mass_kwargs, light_kwargs, eta = gen.__sample_kwargs(random.PRNGKey(i*4+r), i, lens_rotation=r)
                model_s1 = gen.get_model(mass_kwargs, light_kwargs, eta, source=1)
                model_s2 = gen.get_model(mass_kwargs, light_kwargs, eta, source=2)

                lens_img /= np.max(lens_img[70:90, 70:90])
                model_s1 /= 3 * np.max(model_s1)
                model_s2 /= 5 * np.max(model_s2)
                
                plt.imshow(lens_img + model_s1 + model_s2, extent=extent)
                plt.savefig(f"euclid_generator/testing_images/model({i},{r}).png")
                plt.close()

                plt.imshow(lens_img + model_s1 + model_s2, extent=extent, norm="log")
                plt.savefig(f"euclid_generator/testing_images/model({i},{r})_log.png")
                plt.close()



    @staticmethod
    def main():
        # EuclidGenerator.get_final_sim()
        # EuclidGenerator.generate_preso_figures2()
        # EuclidGenerator.big_grid()
        EuclidGenerator.testing1()
        

EuclidGenerator.main()
