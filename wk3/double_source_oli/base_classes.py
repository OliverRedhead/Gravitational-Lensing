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

    def __init__(self, size: int) -> None:
        """
        Initialise a Deflector instance

        Parameters
        ----------
        size : int
            The resolution of the deflector for producing images and lensing.
            Note that the array is always square
        """
        self.size = size
        self.cx = self.cy = (self.size - 1) / 2.0
    

    """getter methods"""

    def get_deflection(self) -> tuple[np.ndarray, np.ndarray]:
        """return the deflection map of the lens"""
        return np.zeros((self.size, self.size)), np.zeros((self.size, self.size))

    def get_deflection_point(self, r):
        return 0, 0

    def get_convergence(self) -> np.ndarray:
        """return the convergence map of the lens"""
        return np.zeros((self.size, self.size))
    
    def get_potential(self) -> np.ndarray:
        """return the potential map of the lens"""
        return np.zeros((self.size, self.size)) 

    def get_magnification(self) -> np.ndarray:
        # Function mapping a single 2D point
        def map_point(r):
            return self.map_jax(r)

        Y, X = np.indices((self.size, self.size))
        R = np.stack((X, Y), axis=-1).astype(np.float32)  # shape (nx, ny, 2)

        jac_fn = jax.jit(jax.vmap(jax.vmap(jax.jacfwd(map_point), in_axes=0), in_axes=0))
        A = jac_fn(R)  # shape (nx, ny, 2, 2)

        mu_inv = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        return np.array(mu_inv)

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
        mapped_curves = [np.array([self.map_jax(point) for point in polygon]) for polygon in curves]
        return mapped_curves


    """class methods"""

    def lens_source(self, source: Source) -> np.ndarray:
        """
        Return the image of the lensed source.
        
        Paramters
        --------
        source : Source
            the image to be lensed. Must be the same shape as lens
        cx : int
            the central x coordinate of the lens
        cy : int
            the central y coordinate of the lens
        """
        
        alpha_x, alpha_y = self.get_deflection()

        Y, X = np.indices((self.size, self.size))

        beta_x = X - alpha_x
        beta_y = Y - alpha_y

        beta_x = np.clip(beta_x, 0, self.size - 1)
        beta_y = np.clip(beta_y, 0, self.size - 1)

        coords = np.array([beta_y.ravel(), beta_x.ravel()])  
        image = map_coordinates(source.array, coords, order=1, mode='nearest').reshape(self.size, self.size)

        return image

    def set_lens_center(self, cx: float, cy: float) -> None:
        self.cx = cx
        self.cy = cy

    def map_jax(self, r) -> jnp.ndarray:
        
        x,y = r[0], r[1]

        alpha_x, alpha_y = self.get_deflection_point(r)

        beta_x = x - alpha_x
        beta_y = y - alpha_y

        return jnp.array([beta_x, beta_y])



    """simulate the resolution, blur and noise of real intruments"""

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



class SIS(Deflector):
        
    def __init__(self, size: int, theta_E: float) -> None:
        super().__init__(size)
        self.theta_E = theta_E


    """getter methods"""

    def get_deflection(self) -> tuple[np.ndarray, np.ndarray]:
        Y, X = np.indices((self.size, self.size))

        Rx = X - self.cx
        Ry = Y - self.cy
        R = np.sqrt(Rx**2 + Ry**2)
        R[R==0] = 1e-6

        alpha_x = self.theta_E * Rx/R 
        alpha_y = self.theta_E * Ry/R 

        return alpha_x, alpha_y

    def get_deflection_point(self, r):
        x, y = r[0], r[1]

        rx = x - self.cx
        ry = y - self.cy
        r = jnp.maximum(np.sqrt(rx**2 + ry**2), 1e-6)

        alpha_x = self.theta_E * rx/r 
        alpha_y = self.theta_E * ry/r

        return alpha_x, alpha_y

    def get_convergence(self) -> np.ndarray:
        """
        A getter method for visualising the surface density of the lens.

        Parameters
        ----------
        source : Source
            A source to set the dimensions of the output convergence field
        """
        Y,X = np.indices((self.size, self.size))

        Rx = X - self.cx
        Ry = Y - self.cy
        theta = np.sqrt(Rx**2 + Ry**2)
        theta[theta == 0] = 1e-6

        kappa = self.theta_E / (2 * theta)

        return kappa
    
    def get_potential(self) -> np.ndarray:
        """
        A getter method for visualising the lensing potential of the lens.

        Parameters
        ----------
        source : Source
            A source to set the dimensions of the output potential field
        """    
        Y, X = np.indices((self.size, self.size))

        Rx = X - self.cx
        Ry = Y - self.cy
        theta = np.sqrt(Rx**2 + Ry**2)
        theta[theta == 0] = 1e-6

        psi = self.theta_E * theta

        return psi
    


class PIEMD(Deflector):

    def __init__(self, 
                 size: int, 
                 *, 
                 theta_E: float | None = None,
                 q: float = 1.0, 
                 s: float = 0.0, 
                 phi: float = 0.0,
                 gamma_1: float = 0.0, 
                 gamma_2: float = 0.0
                 ) -> None:
        super().__init__(size)
        
        if isinstance(theta_E, float):
            self.theta_E = theta_E
        else:
            self.theta_E = 0.2 * self.size
        
        self.q = q
        self.s = s
        self.phi = phi
        self.g1 = gamma_1
        self.g2 = gamma_2


    """getter methods"""

    def get_deflection(self) -> tuple[np.ndarray, np.ndarray]:

        Y, X = np.indices((self.size,self.size))
        Rx = X - self.cx
        Ry = Y - self.cy
        
        # rotate
        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        eps = 1e-12

        Re = np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2)
        Re = np.maximum(Re, eps)

        if abs(1 - self.q) < 1e-6: # in the limit of e -> 1, use special case for isothermal
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

        return alpha_x, alpha_y

    def get_deflection_point(self, r):
        
        x,y = r[0], r[1]

        rx = x - self.cx
        ry = y - self.cy

        Rxp =  jnp.cos(self.phi) * rx + jnp.sin(self.phi) * ry
        Ryp = -jnp.sin(self.phi) * rx + jnp.cos(self.phi) * ry

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

        alpha_x += 2 * self.g1 * rx + 2 * self.g2 * ry
        alpha_y += 2 * self.g2 * rx - 2 * self.g1 * ry

        return alpha_x, alpha_y

    def get_convergence(self) -> np.ndarray:

        Y, X = np.indices((self.size, self.size))

        Rx = X - self.cx
        Ry = Y - self.cy

        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        Re = np.maximum(
            np.sqrt(self.q**2 * (self.s**2 + Rxp**2) + Ryp**2),
            1e-12
        )

        return self.theta_E / (2 * Re)
    
    def get_potential(self) -> np.ndarray:
        Y, X = np.indices((self.size, self.size))
        Rx = X - self.cx
        Ry = Y - self.cy

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

    def get_magnification(self) -> np.ndarray:
        # Function mapping a single 2D point
        def map_point(r):
            return self.map_jax(r)

        Y, X = np.indices((self.size, self.size))
        R = np.stack((X, Y), axis=-1).astype(np.float32)  # shape (nx, ny, 2)

        jac_fn = jax.jit(jax.vmap(jax.vmap(jax.jacfwd(map_point), in_axes=0), in_axes=0))
        A = jac_fn(R)  # shape (nx, ny, 2, 2)

        mu_inv = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        return np.array(mu_inv)

    def get_critical_curves(self):
        a = self.get_magnification()
        fig = plt.figure()
        cs = plt.contour(a, levels=[0])
        plt.close(fig)

        segments = cs.allsegs
        lvl0segs = segments[0]

        return lvl0segs



class Plane:

    def __init__(self, source: Source, deflector: Deflector | None = None) -> None:
        self.source = source
        self.deflector = deflector
