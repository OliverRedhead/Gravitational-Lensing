import numpy as np
import scipy as sp 
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import lax


class Source:

    def __init__(self, array : np.ndarray = np.array([])) -> None:
        self.array = array
        self.colour = array.shape[-1] == 3

    def get_source(self) -> np.ndarray:
        return self.array

    def pad(self, pad):
        arr = self.array

        if arr.ndim == 2:
            # Grayscale image (H, W)
            H, W = arr.shape
            padded = np.zeros((H + 2*pad, W + 2*pad))
            padded[pad:pad+H, pad:pad+W] = arr

        elif arr.ndim == 3:
            # RGB image (H, W, 3)
            H, W, C = arr.shape # type:ignore
            assert C == 3

            padded = np.zeros((H + 2*pad, W + 2*pad, 3))
            padded[pad:pad+H, pad:pad+W, :] = arr
            padded = padded / np.max(padded)

        else:
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB)")

        self.array = padded
        return self.array.shape
    
    def get_indices(self):
        ny, nx, *_ = self.array.shape

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)

        X, Y = np.meshgrid(x,y)

        return X,Y
    
    def plot_source(self):
        plt.imshow(self.array, origin='lower')
        plt.show()

class DiskSource(Source):

    """
    A class which will represent the source in a lensing system.
    
    Specifically represents a uniform disk of light as an `n x n` numpy array.
    """

    def __init__(self, size: int, radius: float):
        """
        intialises a DiskSource instance.

        Parameters
        ----------
        size : int
            the size of the square source array.

        radius : float
            the radius of the uniform disk
        """
        self.size = size
        self.radius = radius
        super().__init__(self._make_disk())

    def _make_disk(self):
        """
        initialiser method which fills the source array with a disk of 
        radius `radius` centered at center of the array
        """
        y, x = np.indices((self.size, self.size))
        cx = cy = (self.size - 1) / 2

        r2 = (x - cx)**2 + (y - cy)**2
        return (r2 <= self.radius**2).astype(float)


class Deflector:

    def __init__(self, source: Source = Source()) -> None:
        self.source = source
        self.ny, self.nx, *_ = self.source.array.shape
        self.image = None

    def generate_image(self) -> np.ndarray:
        self.image = self.source.array
        return self.source.array
    
    def get_convergence(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def get_potential(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def set_source(self, new_source: Source) -> None:
        self.source = new_source
        self.ny, self.nx = self.source.array.shape
        self.image = None

    def get_image(self) -> np.ndarray:
        if not isinstance(self.image, np.ndarray):
            return self.generate_image()
        return self.image


    """ Make (cx,cy) a free parameter """

    def align_lens(self, cx: int = 0, cy: int = 0) -> None:
        """
        sets the pixel given (cx,cy) as the center of the lens
        """
        self.cx = cx
        self.cy = cy
        self.image = None
        return
    
    def align_source(self, cx: int = 0, cy: int = 0) -> None:
        """
        Move the source to (cx, cy) by shifting the lens
        in the opposite direction.
        """
        dx = self.cx - cx
        dy = self.cy - cy

        self.cx += dx
        self.cy += dy

        self.image = None
        return

    
    """ Some helper methods """

    def pad_source(self, pad) -> None:
        self.ny, self.nx, *_ = self.source.pad(pad) 
        self.cx = (self.nx-1) / 2.0
        self.cy = (self.ny-1) / 2.0
        self.image = None

    def clear_image(self) -> None:
        self.image = None

    """ make image look more like real telescope images """

    @staticmethod
    def set_resolution(arr, *, fov=2.0, telescope='hubble') -> np.ndarray:
        """
        static method to convert an array to the desired resolution of a telescope.

        Parameters
        ----------
        arr: np.ndarray
            image to convert resolution
        fov : float
            field of view in arcseconds
        telescope : str
            the key of the desired telescope {'hubble', 'euclid'}
        """
        
        RES = {'hubble': 0.04, 'euclid': 0.1, 'JWST': 0.063}
        
        if telescope not in RES:
            raise NotImplementedError(f"telescope {telescope} not implemented. Please enter either \'hubble\', \'euclid\' or \'JWST\'.")
        
        res = RES[telescope]
        
        ny, nx, *_ = arr.shape
        if nx != ny:
            raise ValueError(f"input array must be square. Not {nx}x{ny}")
        
        pixels = fov / res
        z = pixels / nx
        zoomed_arr = sp.ndimage.zoom(arr, z)
        return zoomed_arr

    @staticmethod
    def blur_image(arr, *, telescope='hubble') -> np.ndarray:
        PSF_FILE = {
            'hubble': 'data/hubble_psf.npy',
            'euclid': 'data/euclid_psf.npy',
            'JWST': 'data/JWST_psf.npy'
            }
        
        if not telescope in PSF_FILE:
            raise NotImplementedError(f"telescope {telescope} not implemented. Please enter either \'hubble\', \'euclid\' or \'JWST\'..")

        psf = np.load("data/psf.npy")
        blur = sp.signal.convolve(arr, psf, mode='same')
        return blur

    @ staticmethod
    def add_noise(arr, *, t=600, telescope='hubble', signal_to_noise=500) -> np.ndarray:
        """
        simulate noise and add it to the input array
        """

        READ_NOISE = {'hubble': 8.7, 'euclid': 4.2, 'JWST': 15.8} # e- rms
        DARK_CURRENT = {'hubble': 4.1e-2, 'euclid': 1.4e-2, 'JWST': 1.9e-3} # e-/s/pixel

        # (arbitrary scale)
        arr = arr / np.max(arr)
        signal_e = arr * signal_to_noise

        # shot noise (Poisson)
        signal_e[signal_e < 0] = 0
        signal_noisy = np.random.poisson(signal_e).astype(float)

        # dark current (Poisson)
        dark_rate = DARK_CURRENT[telescope]
        dark_e = np.random.poisson(dark_rate * t, size=arr.shape).astype(float)
        signal_noisy += dark_e

        # read noise (Gaussian) 
        read_sigma = READ_NOISE[telescope]
        read_noise = np.random.normal(0, read_sigma, size=arr.shape)
        signal_noisy += read_noise

        return signal_noisy




class PIEMD(Deflector):

    """
    Class for calculating and visualising a pseudo-isothermal elliptical mas distribution (PIEMD).
    On initialisation, can specify einstein radius, ellipticity, angle of principle axes and 
    external shear parameters

    Notes
    -----
    - In the limit of e -> 1 we use a different expression for finding deflection angles as the general case. 
    Using small angle approximations, we evaluate the limits.

    - External Shear (XS) does not rotate with the mass distribution. I.e. its orientation is independant of phi. 
    This may be changed in future but for now is how I'm doing it.

    - Mapping coordinate uses bilinear spline interpoltion from scipy.ndimage.map_coordinates. Order = 1 is an arbitrary choice for now
    """

    def __init__(self, source: Source, *, theta_E=None, q=1.0, s=0.0, phi=0.0, gamma_1=0.0, gamma_2=0.0, lens_colour=[1.0,1.0,1.0]) -> None:
        """
        Initialise a pseudo-isothermal elliptical mas distribution (PIEMD) deflector class.
        
        Parameters
        ----------
        theta_E : float
            Einstein radius used to scale the size of the deflector.

        q : float = 1.0
            axis ratio parameter determines how elliptical the resulting convergence map is. 
            A value in (0, 1] where 1 is circular. 
        
        s : float = 0.0
            core parameter. This parameter softens the singularity at the centre of the distribution.

        phi : float = 0.0
            Principal axis angle. This paramater determines the angle of the principal axes of the ellipse to the x and y axes.
            Measured in radians counter-clockwise.

        gamma_1 : float = 0.0
            External shear parameter in xy direction

        gamma_2 : float = 0.0
            External shear parameter in x=y, x=-y direction
        """

        super().__init__(source)
        
        if isinstance(theta_E, float) or isinstance(theta_E, int):
            self.theta_E = theta_E
        else:
            # Default Einstein radius: 0.3 times the larger image dimension
            self.theta_E = 0.3 * max(self.source.array.shape)

        if q > 1 or q <= 0:
            raise ValueError(f"Axis ratio must be in the range (0, 1], not: q={q}")
        
        self.q = q
        self.s = s
        self.phi = phi

        self.g1 = gamma_1
        self.g2 = gamma_2

        ny, nx, *_ = self.source.array.shape

        self.cx = (nx-1) / 2.0
        self.cy = (ny-1) / 2.0

        self.colour = source.colour
        self.lens_colour = np.array(lens_colour)

        self.image = None

    def generate_image(self, 
                       *, 
                       R_e: float|None = None, 
                       lens: str = 'sersic_core',
                       sigma: float|None = None,
                       I_lens: float = 1.0
                       ) -> np.ndarray:

        src = self.source.array
        nx, ny, nc = self.nx, self.ny, 1

        if src.ndim == 2:
            ny, nx = src.shape
        if src.ndim == 3: # if colour
            ny, nx, nc = src.shape # type:ignore

        cx = self.cx
        cy = self.cy


        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy
        
        # rotate
        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        eps = 1e-12

        Re = np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2)
        Re = np.maximum(Re, eps)

        if abs(1 - self.q) < 1e-6: # in the limit of e -> 1, use sepcial case for isothermal
            rc = np.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_x = self.theta_E * Rxp / (rc + self.s + eps) \
                + 2 * self.g1 * Rx + 2 * self.g2 * Ry
            alpha_y = self.theta_E * Ryp / (rc + self.s + eps) \
                + 2 * self.g2 * Rx - 2 * self.g1 * Ry 
        
        else:
            a = np.sqrt(1 - self.q**2)

            denom_x = np.maximum(Re + self.s, eps)
            denom_y = np.maximum( Re + self.q**2 * self.s, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y

            # clip to valid domain for arctanh
            u_y = np.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_xp = self.theta_E * (1 / a) * np.arctan(u_x)
            alpha_yp = self.theta_E * (1 / a) * np.arctanh(u_y)

            # rotate deflection back to image (x,y) coords (inverse rotation)
            alpha_x_rot = np.cos(self.phi) * alpha_xp - np.sin(self.phi) * alpha_yp
            alpha_y_rot = np.sin(self.phi) * alpha_xp + np.cos(self.phi) * alpha_yp

            # add external shear in image coords
            alpha_x = alpha_x_rot + 2 * self.g1 * Rx + 2 * self.g2 * Ry
            alpha_y = alpha_y_rot + 2 * self.g2 * Rx - 2 * self.g1 * Ry 
            

        beta_x = X - alpha_x
        beta_y = Y - alpha_y
        
        # using bilinear (order=1) spline interpolation - an arbitrary choice for now
        coords = np.array([beta_y.ravel(), beta_x.ravel()])  

        if nc == 1:
            image = map_coordinates(src, coords, order=1, mode='nearest').reshape(ny, nx)
        else:
            image = np.zeros((ny, nx, nc), dtype=src.dtype)
            for c in range(nc):
                image[:, :, c] = map_coordinates(src[:, :, c], coords, order=1, mode='nearest').reshape(ny, nx)

        if self.colour:
            Imax = np.max(image)
            if(Imax > 0):
                image = image / np.max(image)
            Imax = np.clip(image, 0.0, 1.0) 

        self.image = image

        if lens == 'sersic_core':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            image += I_lens * self.generate_sersic_core_lens(
                    n=4,
                    R_e=R_e,
                    R_b=self.s,
                    gamma=0.1,
                    alpha=30
                )
            
        elif lens == 'sersic':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            image += I_lens * self.generate_sersic_lens(
                    n=4,
                    R_e=R_e
                )
        
        elif lens == 'gaussian':
            if not isinstance(sigma, float):
                sigma = 1/3 * self.theta_E

            image += I_lens * self.generate_gaussian_lens(n=2, sigma=sigma)
        
        elif not lens == 'none':
            raise NotImplementedError(f"Lens {lens} not implemented. Please enter either \'sersic\', \'sersic_core\', \'gaussian\' or \'none\'.")
        
        return image


    """ Different lensing galaxy distributions """

    def generate_gaussian_lens(self, sigma, n) -> np.ndarray:
        """
        Generate a 2D elliptical Gaussian representing a lens galaxy.

        Parameters
        ----------
        shape : tuple of int
            (ny, nx) size of the output image.
        einstein_radius : float
            Characteristic radius (in pixels) of the lens (similar to sigma for Gaussian).
        axis_ratio : float
            Minor-to-major axis ratio q = b/a (0 < q <= 1).
        I0 : float
            Peak intensity.

        Returns
        -------
        np.ndarray
            2D image of the elliptical Gaussian lens.
        """
        nx, ny = self.nx, self.ny
        cx, cy = self.cx, self.cy

        Y, X = np.indices((ny, nx))

        # Center coordinates
        dx = X - cx
        dy = Y - cy

        # Rotate coordinates
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        X_rot = cos_phi * dx + sin_phi * dy
        Y_rot = -sin_phi * dx + cos_phi * dy

        # Elliptical radius
        r2 = (self.q * X_rot)**2 + Y_rot**2
        r2 = r2**(n/2)  

        # Elliptical Gaussian
        img = np.exp(-0.5 * r2 / sigma**2)

        return img

    def generate_sersic_lens(self, n: float, R_e: float) -> np.ndarray:
        """
        Generate a 2D elliptical Sersic representing a lens galaxy.

        Parameters
        ----------
        n : float
            shape parameter controlling the overall curvature of the lens
        R_E : float
            the scale radius (half light radius)

        Returns
        -------
        img : np.ndarray
            an image of a lens according to input profile
        """
        def b(n):
            bn = 2 * n - 1/3
            
            if n > 8:
                return bn
            
            if n > 0.36:
                a1 = 4/(405 * n)
                a2 = 46/(25515 * n**2)
                a3 = 131/(1148175 * n**3)
                a4 = 2194697/(30690717750 * n**4)
                return bn + a1 + a2 + a3 - a4
            
            else:
                raise ValueError(f"n must be in range (0.36, infty). Not {n}")

        nx, ny = self.nx, self.ny
        cx, cy = self.cx, self.cy

        Y, X = np.indices((ny, nx))

        # Center coordinates
        dx = X - cx
        dy = Y - cy

        # Rotate coordinates
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        X_rot = cos_phi * dx + sin_phi * dy
        Y_rot = -sin_phi * dx + cos_phi * dy

        # Elliptical radius
        R = np.sqrt((self.q * X_rot)**2 + Y_rot**2)

        img = np.exp(-b(n) * ( (R/R_e)**(1/n) - 1))
        return img
    
    def generate_sersic_core_lens(self, n: float, R_e: float, R_b: float, gamma: float, alpha: float) -> np.ndarray:
        """
        Generate a 2D elliptical Sersic representing a lens galaxy.

        Parameters
        ----------
        n : float
            shape parameter controlling the overall curvature of the lens
        R_E : float
            the scale radius (half light radius). As we have used the same function for bn as the sersic case,
            this radius will not be 100% accurate.
        R_b : float
            The break radius R_b is the point at which the profile changes from one regime to another.
            We will use the self.s core parameter to set this in practice.
        gamma : float
            The slope of the inner power-law region. Usually we want this less than n, usually about 0.1-0.3
        alpha : float
            Controls the sharpness of the transition between the cusp and the outer Sersic profile

        Returns
        -------
        img : np.ndarray
            an image of a lens according to input profile
        """
        def b(n:float):
            bn = 2 * n - 1/3
            
            if n > 8:
                return bn
            
            if n > 0.36:
                a1 = 4/(405 * n)
                a2 = 46/(25515 * n**2)
                a3 = 131/(1148175 * n**3)
                a4 = 2194697/(30690717750 * n**4)
                return bn + a1 + a2 + a3 - a4
            
            else:
                raise ValueError(f"n must be in range (0.36, infty). Not {n}")

        nx, ny = self.nx, self.ny
        cx, cy = self.cx, self.cy

        Y, X = np.indices((ny, nx))

        # Center coordinates
        dx = X - cx
        dy = Y - cy

        # Rotate coordinates
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        X_rot = cos_phi * dx + sin_phi * dy
        Y_rot = -sin_phi * dx + cos_phi * dy

        # Elliptical radius
        R = np.maximum(np.sqrt((self.q * X_rot)**2 + Y_rot**2), 1e-6)

        bn = b(n)
        I_prime = 2**(-gamma/alpha) * np.exp(bn * 2**(1/(alpha*n)) * (R_b/R_e)**(1/n))
        a1 = 1 + (R_b/R)**alpha
        a2 = (R**alpha + R_b**alpha)/(R_e**alpha)
        
        img = I_prime * a1**(gamma/alpha) * np.exp( -bn * a2**(1/(n*alpha)) )

        return img


    """ working on telescope images """

    def generate_hubble_image(self, 
                              *, 
                              R_e: float|None = None, 
                              lens='sersic_core', 
                              sigma: float|None = None,
                              I_lens: float = 1.0
                              ):
        """
        Generate an image with lensing galaxy and make it look like it was taken by euclid.
        """
        if not isinstance(self.image, np.ndarray):
            self.image = self.generate_image(lens='none')

        img = self.image.astype(float) 
        img /= np.max(img)

        # --- add lens --- #
        if lens == 'sersic_core':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_core_lens(
                    n=4,
                    R_e=R_e,
                    R_b=self.s,
                    gamma=0.1,
                    alpha=30
                )
            
        elif lens == 'sersic':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_lens(
                    n=4,
                    R_e=R_e
                )
        
        elif lens == 'gaussian':
            if not isinstance(sigma, float):
                sigma = 1/3 * self.theta_E

            img += I_lens * self.generate_gaussian_lens(n=2, sigma=sigma)

        elif lens != 'none':
            raise NotImplementedError(f"Lens {lens} not implemented. Please enter either \'sersic\', \'sersic_core\', \'gaussian\' or \'none\'.")
        
        # --- zoom image to required resolution -- #
        img_zoomed = self.set_resolution(img, telescope='hubble')

        # --- blur image with euclid psf -- #
        img_blurred = self.blur_image(img_zoomed, telescope='hubble')

        # -- add noise -- #
        img_noise = self.add_noise(img_blurred, telescope='hubble')

        return img_noise

    def generate_euclid_image(self, 
                              *, 
                              R_e: float|None = None, 
                              lens='sersic_core', 
                              sigma: float|None = None,
                              I_lens: float = 1.0
                              ):
        """
        Generate an image with lensing galaxy and make it look like it was taken by euclid.
        """
        if not isinstance(self.image, np.ndarray):
            self.image = self.generate_image(lens='none')

        img = self.image.astype(float) 
        img /= np.max(img)

        # --- add lens --- #
        if lens == 'sersic_core':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_core_lens(
                    n=4,
                    R_e=R_e,
                    R_b=self.s,
                    gamma=0.1,
                    alpha=30
                )
            
        elif lens == 'sersic':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_lens(
                    n=4,
                    R_e=R_e
                )
        
        elif lens == 'gaussian':
            if not isinstance(sigma, float):
                sigma = 1/3 * self.theta_E

            img += I_lens * self.generate_gaussian_lens(n=2, sigma=sigma)

        elif lens != 'none':
            raise NotImplementedError(f"Lens {lens} not implemented. Please enter either \'sersic\', \'sersic_core\', \'gaussian\' or \'none\'.")
        
        # --- zoom image to required resolution -- #
        img_zoomed = self.set_resolution(img, telescope='euclid')

        # --- blur image with euclid psf -- #
        img_blurred = self.blur_image(img_zoomed, telescope='euclid')

        # -- add noise -- #
        img_noise = self.add_noise(img_blurred, telescope='euclid')

        return img_noise

    def generate_JWST_image(self, 
                              *, 
                              R_e: float|None = None, 
                              lens='sersic_core', 
                              sigma: float|None = None,
                              I_lens: float = 1.0
                              ):
        """
        Generate an image with lensing galaxy and make it look like it was taken by JWST.
        """
        if not isinstance(self.image, np.ndarray):
            self.image = self.generate_image(lens='none')

        img = self.image.astype(float) # type:ignore
        img /= np.max(img)

        # --- add lens --- #
        if lens == 'sersic_core':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_core_lens(
                    n=4,
                    R_e=R_e,
                    R_b=self.s,
                    gamma=0.1,
                    alpha=30
                )
            
        elif lens == 'sersic':
            if not isinstance(R_e, float):
                R_e = np.random.uniform(0.2*self.theta_E, 0.4*self.theta_E)

            img += I_lens * self.generate_sersic_lens(
                    n=4,
                    R_e=R_e
                )
        
        elif lens == 'gaussian':
            if not isinstance(sigma, float):
                sigma = 1/3 * self.theta_E

            img += I_lens * self.generate_gaussian_lens(n=2, sigma=sigma)

        elif lens != 'none':
            raise NotImplementedError(f"Lens {lens} not implemented. Please enter either \'sersic\', \'sersic_core\', \'gaussian\' or \'none\'.")
        
        # --- zoom image to required resolution -- #
        img_zoomed = self.set_resolution(img, telescope='JWST')

        # --- blur image with JWST psf -- #
        img_blurred = self.blur_image(img_zoomed, telescope='JWST')

        # -- add noise -- #
        img_noise = self.add_noise(img_blurred, telescope='JWST')

        return img_noise



    """ two mapping methods - one for normal mapping and one for jax. They should both work the exact same way. """

    def map(self, r):
        """
        given a coordinate on image plane (x,y), returns corresponding point on source plane.
        """
        
        x, y = r

        cx = self.cx
        cy = self.cy

        # image->relative coordinates
        Rx = x - cx
        Ry = y - cy

        # rotate into lens principal axes
        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        eps = 1e-12
        Re = np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2)
        Re = max(Re, eps)

        if abs(1 - self.q) < 1e-6:
            rc = np.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_xp = self.theta_E * Rxp / (rc + self.s + eps)
            alpha_yp = self.theta_E * Ryp / (rc + self.s + eps)
        else:
            a = np.sqrt(max(0.0, 1 - self.q**2))
            denom_x = max(Re + self.s, eps)
            denom_y = max(Re + self.q**2 * self.s, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y
            u_y = np.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_xp = self.theta_E * (1 / a) * np.arctan(u_x)
            alpha_yp = self.theta_E * (1 / a) * np.arctanh(u_y)

        # rotate deflection back to image coords
        alpha_x = np.cos(self.phi) * alpha_xp - np.sin(self.phi) * alpha_yp
        alpha_y = np.sin(self.phi) * alpha_xp + np.cos(self.phi) * alpha_yp

        # add external shear (defined in image coords)
        alpha_x += 2 * self.g1 * Rx + 2 * self.g2 * Ry
        alpha_y += 2 * self.g2 * Rx - 2 * self.g1 * Ry

        beta_x = x - alpha_x
        beta_y = y - alpha_y

        return beta_x, beta_y
    
    def map_jax(self, r):
        
        x,y = r[0], r[1]
        nx, ny, *_ = self.nx, self.ny

        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

        cx = self.cx
        cy = self.cy

        Rx = x - cx
        Ry = y - cy

        Rxp =  jnp.cos(self.phi) * Rx + jnp.sin(self.phi) * Ry
        Ryp = -jnp.sin(self.phi) * Rx + jnp.cos(self.phi) * Ry

        eps = 1e-12
        Re = jnp.maximum(jnp.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2), eps)

        def circular_case(_):
            rc = jnp.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_xp = self.theta_E * Rxp / (rc + self.s + eps)
            alpha_yp = self.theta_E * Ryp / (rc + self.s + eps)
            return alpha_xp, alpha_yp
        
        def elliptical_case(_):
            a = jnp.sqrt(jnp.maximum(0.0, 1 - self.q**2))
            denom_x = jnp.maximum(Re + self.s, eps)
            denom_y = jnp.maximum(Re + self.q**2 * self.s, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y
            u_y = jnp.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_xp = self.theta_E * (1.0 / a) * jnp.arctan(u_x)
            alpha_yp = self.theta_E * (1.0 / a) * jnp.arctanh(u_y)
            return alpha_xp, alpha_yp

        alpha_xp, alpha_yp = lax.cond(
            jnp.abs(1 - self.q) < 1e-6,
            circular_case,
            elliptical_case,
            operand=None,
        )

        alpha_x =  jnp.cos(self.phi) * alpha_xp - jnp.sin(self.phi) * alpha_yp
        alpha_y =  jnp.sin(self.phi) * alpha_xp + jnp.cos(self.phi) * alpha_yp

        alpha_x += 2 * self.g1 * Rx + 2 * self.g2 * Ry
        alpha_y += 2 * self.g2 * Rx - 2 * self.g1 * Ry

        beta_x = x - alpha_x
        beta_y = y - alpha_y

        return jnp.array([beta_x, beta_y])


    """ getter methods for testing """

    def get_convergence(self):
        nx, ny = self.nx, self.ny

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        cx = self.cx
        cy = self.cy

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy

        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        Re = np.maximum(
            np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2),
            1e-12
        )

        return self.theta_E / (2 * Re)

    def get_potential(self):
        nx, ny = self.nx, self.ny

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        cx = self.cx
        cy = self.cy

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy

        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        Re = np.maximum(
            np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2),
            1e-12
        )

        Rf = np.sqrt((1-self.q**2) * Rxp**2 + (Re + self.s)**2) 

        eps = 1e-12
        if abs(1 - self.q) < 1e-6:
            rc = np.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_x = self.theta_E * Rxp / (rc + self.s + eps)
            alpha_y = self.theta_E * Ryp / (rc + self.s + eps)
        
        else:
            a = np.sqrt(1 - self.q**2)

            denom_x = np.maximum(Re + self.s, eps)
            denom_y = np.maximum(Re + self.q**2 * self.s, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y

            # clip to valid domain for arctanh ( for floating point errors)
            u_y = np.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_x = self.theta_E * (1 / a) * np.arctan(u_x)
            alpha_y = self.theta_E * (1 / a) * np.arctanh(u_y)

        # NOTE external shear (psi_2) does NOT rotate with the mass distribution
        psi_1 =  Rxp * alpha_x + Ryp * alpha_y - self.s * np.log(Rf) 
        psi_2 = self.g1 * (Rx**2 - Ry**2) + 2 * self.g2 * Rx * Ry       # external shear 

        return psi_1 + psi_2

    def get_magnification(self):

        def map_only_r(r):
            return self.map_jax(r)

        jac_map = jax.jacfwd(map_only_r)

        # Optional but highly recommended:
        jac_map = jax.jit(jac_map)

        X,Y = self.source.get_indices()
        R = np.stack((X, Y), axis=-1).astype(np.float32)   # shape (nx, ny, 2)
        R_flat = R.reshape(-1, 2)

        map_per_point = lambda r: self.map_jax(r)
        A_flat = jax.vmap(jax.jacfwd(map_per_point))(R_flat)
        A = A_flat.reshape(X.shape[0], X.shape[1], 2, 2)

        a = A[...,0,0]*A[...,1,1] - A[...,1,0]*A[...,0,1] # inverse magnification
        return a
    
    def get_critical_curves(self):
        a = self.get_magnification()
        fig = plt.figure()
        cs = plt.contour(a, levels=[0])
        plt.close(fig)

        segments = cs.allsegs
        lvl0segs = segments[0]

        return lvl0segs
    
    def get_caustics(self):
        curves = self.get_critical_curves()
        mapped_curves = [np.array([self.map(point) for point in polygon]) for polygon in curves]
        return mapped_curves

    def get_image(self):
        if isinstance(self.image, np.ndarray):
            return self.image
        
        return self.generate_image()


    """ These methods help me view things """

    def plot_source(self, *, critical_curves=False, caustics=False, 
                    convergence=False, potential=False, image=True, noise=False):


        # colour
        if self.source.array.shape[-1] == 3:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.source.array

            image_arr = np.clip(image_arr, 0.0, 1.0)

            plt.imshow(image_arr, origin='lower')

        else:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.source.array

            cbar = plt.imshow(image_arr, cmap='inferno', origin='lower')
            plt.colorbar(cbar)

        if convergence:
            plt.contour(self.get_convergence())

        if potential:
            plt.contour(self.get_potential())

        if critical_curves:
            for curve in self.get_critical_curves():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='red')

        if caustics:
            for curve in self.get_caustics():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='orange')

        plt.show()

    def plot_image(self, *, critical_curves=False, caustics=False, 
                convergence=False, potential=False, image=True, lens=True, I_0=2.0):


        # colour
        if self.source.array.shape[-1] == 3:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.generate_image()

            if lens:
                image_arr += self.generate_gaussian_lens(sigma=self.theta_E, n=2)

            image_arr = np.clip(image_arr, 0.0, 1.0)
            plt.imshow(image_arr, origin='lower')

        # greyscale
        else:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                if not isinstance(self.image, np.ndarray):
                    image_arr += self.generate_image()
                else:
                    image_arr = self.image
            

            if lens:
                image_arr += self.generate_gaussian_lens(sigma=self.theta_E, n=2)

            cbar = plt.imshow(image_arr, cmap='inferno', origin='lower')
            plt.colorbar(cbar)

        if convergence:
            plt.contour(self.get_convergence())

        if potential:
            plt.contour(self.get_potential())

        if critical_curves:
            for curve in self.get_critical_curves():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='red')

        if caustics:
            for curve in self.get_caustics():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='orange')

        plt.show()

    
        

        