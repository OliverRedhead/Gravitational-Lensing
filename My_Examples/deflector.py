import numpy as np
from numpy import fft
from astropy.io import fits


class Deflector(object):
    """
    This class represents the deflection of a mass distribution. 
    Given in input surface density (convergence) map "filekappa", this class can calculate the direction and magnitude of deflection.
    """

    def __init__(self, filekappa, pad=False):
        self.kappa,self.header = fits.getdata(filekappa, header=True)
        self.nx, self.ny = self.kappa.shape
        if(pad):
            self.pad()
        self.kx, self.ky = self.kernel()


    def kernel(self):
        x = np.linspace(-0.5, 0.5, self.nx)
        y = np.linspace(-0.5, 0.5, self.ny)

        print(x)

        kx = fft.fftfreq(self.nx)   # frequencies in cycles per pixel
        ky = fft.fftfreq(self.ny)

        kx, ky = np.meshgrid(x, y)
        norm = kx**2 + ky**2 + 1e-12
        kx, ky = kx/norm, ky/norm

        return kx, ky

    
    def pad(self):
        """
        pads boundaries with zeros to make them not periodic - this is a product of the fft
        """
        def padwithzeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        
        self.kappa = np.lib.pad(self.kappa, self.kappa.shape[0], padwithzeros)

    def deflection_map(self):
        """
        convolves the kernel K and convergence kappa via fft
        """
        # perform fftt
        kappa_fft = fft.rfftn(self.kappa, axes=(0,1))
        kx_fft= fft.rfftn(self.kx, axes=(0,1), s=self.kappa.shape)
        ky_fft= fft.rfftn(self.ky, axes=(0,1), s=self.kappa.shape)

        # convolve
        alphafft_x = kappa_fft * kx_fft
        alphafft_y = kappa_fft * ky_fft

        # inverse fft (1/pi comes from equation in meneghetti lectues: section 2.5.2)
        alpha_x = 1/np.pi * fft.irfftn(alphafft_x)
        alpha_y = 1/np.pi * fft.irfftn(alphafft_y)

        return alpha_x, alpha_y
    
    def image(self):
        return self.kappa
        
