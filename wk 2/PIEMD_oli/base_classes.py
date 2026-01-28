import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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

    def get_image(self) -> np.ndarray:
        return self.source.array
    
    def get_convergence(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def get_potential(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def set_source(self, new_source: Source):
        self.source = new_source
        self.ny, self.nx = self.source.array.shape

    def align_source(self, cx: int = 0, cy: int = 0, source=False ):
        """
        sets the pixel given (cx,cy) as the center of the lens
        """
        self.cx = cx
        self.cy = cy
        return
    
    def pad_source(self, pad):
        self.ny, self.nx, *_ = self.source.pad(pad) 
        self.cx = (self.nx-1) / 2.0
        self.cy = (self.ny-1) / 2.0

    @staticmethod
    def hubble_resolution(arr, * , fov=2.0, res=0.04):
        """
        Takes an array and reduces/increases its resolution 
        to that of the hubble space telescope at 2'' x 2''.

        Parameters
        ----------
        arr : np.ndarray
            A 2d array representing the image to be resolved.
        
        field : float = 2.0
            a float representing the field of view of the image in arcseconds/
            Default is 2.0 arcseconds.

        res : float = 0.04
            the resolution of the camera in arcseconds per pixel.
            Default is 0.04 arcseconds.


        Note
        ----
        - The input array must be square
        - Using the fact that UV/visible resolution is 0.04'' per pixel
        hence resulting image must be 50x50
        """

        ny, nx, *_ = arr.shape
        if nx != ny:
            raise ValueError(f"input array must be square. Not {nx}x{ny}")
        
        pixels = fov / res
        z = pixels / nx
        zoomed_arr = sp.ndimage.zoom(arr, z)
        return zoomed_arr

    @staticmethod
    def hubble_noise(arr, t=600.0, gain=1, dark_level="low"):
        """
        Apply a realistic HST CCD noise model to a normalized image.

        Parameters
        ----------
        arr : np.ndarray
            Input image normalized 0 -> 1

        t : float
            Exposure time in seconds. Default 600s

        gain : int
            e-/DN (1 or 4)

        dark_level : str
            "low", "medium", or "high"

        wavelength : int
            Wavelength in angstroms. Default 6000 Å

        inner_full_well : int
            Full well electrons for inner detector pixels

        outer_full_well : int
            Full well electrons for outer detector pixels

        pixel_mask : np.ndarray, optional
            Boolean mask: True for inner pixels, False for outer

        Note
        ----
        All data is taken from
        https://hst-docs.stsci.edu/stisihb/chapter-7-feasibility-and-detector-performance/7-2-the-ccd
        """
        
        READ_NOISE = {1: 6.2, 4: 8.7}  # e- RMS
        DARK_CURRENT = {"low": 2.9e-2, "medium": 3.4e-2, "high": 4.1e-2}  # e-/s/pix

        # (arbitrary scale)
        arr = arr / np.max(arr)
        signal_e = arr * 500

        # shot noise (Poisson)
        signal_e[signal_e < 0] = 0
        signal_noisy = np.random.poisson(signal_e).astype(float)

        # dark current (Poisson)
        dark_rate = DARK_CURRENT[dark_level]
        dark_e = np.random.poisson(dark_rate * t, size=arr.shape).astype(float)
        signal_noisy += dark_e

        # read noise (Gaussian) 
        read_sigma = READ_NOISE[gain]
        read_noise = np.random.normal(0, read_sigma, size=arr.shape)
        signal_noisy += read_noise

        return signal_noisy

    @staticmethod
    def hubble_blur(arr, fwhm=0.07, res=0.04):
        """
        Convolve an image with the Hubble PSF to simulate telescope blur.

        Parameters
        ----------
        image : np.ndarray
            Input image (2D grayscale).
        fwhm : float
            FWHM of hubble telescope in arcseconds

        Returns
        -------
        np.ndarray
            Blurred image of the same shape as input.
        """

        size = arr.shape[0]
        # arbitrary
        fwhm_pixels = fwhm / res
        sigma = fwhm_pixels / 2.355

        x = np.arange(0, size, 1, float) - size//2
        y = x[:, np.newaxis]
        psf = np.exp(-(x**2 + y**2)/(2*sigma**2))
        psf =  psf / np.sum(psf)

        blur = sp.signal.convolve(arr, psf, mode='same')
        return blur