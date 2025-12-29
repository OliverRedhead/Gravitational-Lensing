import numpy as np
from numpy import fft
from astropy.io import fits
import astropy.constants as cons
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

class Plane:

    # intialiser methods #

    def __init__(self, data: str | np.ndarray, pad: bool = False, padwidth: float = 0.5) -> None:
        """
        Initialise a Plane object

        Parameters
        ----------
        data : str | np.ndarray
            Either a filepath to a .fits data file or a 2D numpy array of data

        Notes
        -----
        This class can be extended to process a variety of filetypes in the future
        """
        if isinstance(data, str): 
            # NOTE can extend this to work on other filetypes
            self.data, _ = fits.getdata(data, header=True)  # type: ignore
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(f"Invalid type for data: {type(data)}. Must be str or np.ndarray.")
        

        self.nopad_nx, self.nopad_ny = self.data.shape
        self.padwidth = padwidth
        if pad:
            self.pad()
        self.nx, self.ny = self.data.shape

    # class methods #

    def pad(self) -> None:
        """
        Pad plane data with zeros.

        This ensures that the boundaries are non-periodic for Fourier transforms,
        which prevents wrap-around artifacts in the calculation.

        Notes
        -----
        The amount of padding is determined by `self.padwidth`, which is a fraction
        of the map size in each dimension.
        """
        px, py = self.kappa.shape

        pad_x = int(px * self.padwidth)
        pad_y = int(py * self.padwidth)
        
        self.kappa = np.pad(
            self.kappa,
            ((pad_x, pad_x), (pad_y, pad_y)),
            mode='constant',
            constant_values=0
        )


    def data_crop(self, arr : np.ndarray):
        """
        Given an array `arr`, this method returns the unpadded version based on the size of the 
        original convergence map.

        Returns
        -------
        cropped_data : ndarray
            The 2D array representing the deflection map of the original section of space.
        """
        px, py = self.data.shape

        pad_x = int(px * self.padwidth)//2
        pad_y = int(py * self.padwidth)//2

        data_crop = arr[pad_y:pad_y+self.nopad_ny,pad_x:pad_x+self.nopad_nx]
        return data_crop
    

    def plot_data(self, *, title="", cbar=True, cmap='viridis') -> None:
        """Plots Plane data"""
        cbar = plt.imshow(self.data, cmap='viridis', origin='lower')
        if cbar:
            plt.colorbar(cbar)
        plt.title(title)
        plt.show()


    def get_data(self) -> np.ndarray:
        """Returns Plane data"""
        return self.data




class Lens(Plane):

    # initialiser methods #

    def __init__(self, data: str | np.ndarray, pad: bool = False, padwidth: float = 0.5):
        """
        Initialize a Deflector object from a convergence map.

        Parameters
        ----------
        filekappa : str
            Filename or path to the convergence map (FITS file).
        pad : bool, optional
            If True, pad the convergence map boundaries (default is False).
        padwidth : float, optional
            Width of padding applied to the map if `pad` is True (default is 0.5).
        """
        super().__init__(data, pad, padwidth)

        self.kappa = self.data
        self.kx, self.ky = self.kernel()

    # class methods #

    def kernel(self):
        """
        Initialize the kernel array used for convolution with the convergence map.

        The kernel is a vector-valued function:

            K(x) = x / |x|^2 

        where `x` is the 2D position vector in the map.

        Notes
        -----
        - The size of the kernel matches the (optionally padded) convergence map.
        - This ensures that the kernel and the map can be convolved using
        element-wise multiplication in Fourier space.
        """
        kx = fft.fftfreq(self.nx).reshape(-1,1)
        ky = fft.fftfreq(self.ny).reshape(1,-1)

        den = kx**2 + ky**2
        den += 1e-12

        kx_kernel = -1j * kx / den
        ky_kernel = -1j * ky / den

        return kx_kernel, ky_kernel


    def deflection_map(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the deflection angle maps from the convergence map.

        The deflection is a convolution of the kernel and the convergence map (see Meneghetti notes).
        We use NumPy's fft library to convolve the two and compute deflection.

        Returns
        -------
        alpha_x : ndarray
            Deflection in the x-direction for each pixel [radians].
        alpha_y : ndarray
            Deflection in the y-direction for each pixel [radians].

        Notes
        -----
        - The FFT-based computation assumes periodic boundary conditions, 
        so you may want to pad the map using `self.pad()` to reduce wrap-around artifacts.
        - The returned arrays are real-valued; the imaginary part from FFT/IFFT is discarded.
        - The kernel is already in fourier space.
        """
        kappa_fft = fft.fft2(self.kappa)

        alpha_x_fft = kappa_fft * self.kx
        alpha_y_fft = kappa_fft * self.ky

        alpha_x = fft.ifft2(alpha_x_fft).real / np.pi
        alpha_y = fft.ifft2(alpha_y_fft).real / np.pi

        return alpha_x, alpha_y



class RayTracer:

    # initialiser methdos #

    def __init__(self, lens: Lens) -> None:
        """
        Given a Lens object `lens`, initialise a RayTracer object

        Parameters
        ----------
        lens : Lens
            A lens we want to ray trace through.

        Notes
        -----
        i. We use the index notation i,j in this class where i corresponds to the x-axis and j the y-axis
        ii. We assume all images are square
        """
        
        self.lens = lens
        self.source = self.trace()

    # class methods

    def trace(self, ndown=1):
        """
        Using the lens equation, trace the lens plane back to the source plane

        Parameters 
        ----------
        ndown : float
            Default to 1. This will reduce the definition of the tracing for visualisation.
        """

        # get deflection angles alpha(x)
        angi, angj = self.lens.deflection_map() 

        # generate grid of position vectors 
        npix = angi.shape[0] # NOTE assuming square image
        npix_ = npix//ndown

        i_range = np.linspace(0,1,npix_) * (npix-1) 
        j_range = np.linspace(0,1,npix_) * (npix-1) 

        i,j = np.meshgrid(i_range, j_range)

        # interpolate (have to reshape first)
        i_ = i.reshape(i.size)
        j_ = j.reshape(j.size)

        angi_ = map_coordinates(angi, [[j],[i]], order=1)
        angj_ = map_coordinates(angj, [[j],[i]], order=1)

        # reshape angles back to mesh
        angi = angi_.reshape( (npix_, npix_) ) 
        angj = angj_.reshape( (npix_, npix_) ) 

        # apply lens equation
        self.i = i
        self.j = j
        self.yi = i - angi
        self.yj = j - angj

    
    def plot_trace(self):
        """
        Scatter plot of lens-plane coordinates (i,j) and mapped source-plane coordinates (yi,yj).
        Creates two subplots side-by-side. If trace() has not been run yet, it will be executed.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(self.i, self.j, s=1, alpha=0.7, color="C0")
        axes[0].set_title("Lens plane (i, j)")
        axes[0].set_xlabel("i (pixel)")
        axes[0].set_ylabel("j (pixel)")
        axes[0].set_aspect("equal", adjustable="box")

        axes[1].scatter(self.yi, self.yj, s=1, alpha=0.7, color="C1")
        axes[1].set_title("Source plane (yi, yj)")
        axes[1].set_xlabel("yi (pixel)")
        axes[1].set_ylabel("yj (pixel)")
        axes[1].set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()