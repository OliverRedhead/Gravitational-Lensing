import numpy as np
from numpy import fft
from astropy.io import fits

class Deflector:
    def __init__(self, filekappa, pad=False, padwidth=0.5):
        self.kappa, self.header = fits.getdata(filekappa, header=True)
        if pad:
            self.pad(padwidth)
        self.nx, self.ny = self.kappa.shape
        self.kx, self.ky = self.kernel()

    def pad(self, padwidth):
        px, py = self.kappa.shape

        pad_x = int(px * padwidth)
        pad_y = int(py * padwidth)
        
        self.kappa = np.pad(
                self.kappa,
                ((pad_x, pad_x), (pad_y, pad_y)),
                mode='constant',
                constant_values=0
            )


    def kernel(self):
        kx = fft.fftfreq(self.nx).reshape(-1,1)
        ky = fft.fftfreq(self.ny).reshape(1,-1)

        den = kx**2 + ky**2
        den += 1e-12

        kx_kernel = -1j * kx / den
        ky_kernel = -1j * ky / den

        return kx_kernel, ky_kernel

    def deflection_map(self):
        kappa_fft = fft.fft2(self.kappa)

        alpha_x_fft = kappa_fft * self.kx
        alpha_y_fft = kappa_fft * self.ky

        alpha_x = fft.ifft2(alpha_x_fft).real / np.pi
        alpha_y = fft.ifft2(alpha_y_fft).real / np.pi

        return alpha_x, alpha_y

    def image(self):
        return self.kappa
