import numpy as np
from base_classes import Source, Deflector


class EinsteinDeflector(Deflector):

    """
    A class to calculate the deflection of an isothermal sphere.
    """

    def __init__(self, theta_E : float) -> None:
        """
        intialises a Deflector instance

        Parameters
        ----------
        theta_E : float
            A float to specify the Einstein radius of the respective deflector
        """
        self.theta_E = theta_E

    def get_image(self, source : Source):
        """
        Calculates the deflection of the source given the Einstein radius of the lens using the lensing equation.

        Parameters
        ----------
        source : DiskSource
            The source plane to be lensed
        """

        src = source.get_source()
        image = np.zeros_like(src)
        nx, ny = src.shape

        cx = (nx - 1)/2
        cy = (ny - 1)/2

        for y in range(ny):
            for x in range(nx):

                im_val = src[y,x]
                if(im_val == 0):
                    continue

                rx = x - cx
                ry = y - cy
                r = np.sqrt(rx**2 + ry**2)
                if(r == 0):
                    r = 1e-6
                
                alpha_x = self.theta_E * rx / r
                alpha_y = self.theta_E * ry / r

                beta_x = round(x - alpha_x)
                beta_y = round(y - alpha_y)

                image[beta_y, beta_x] += im_val

        self.image = image
        return image

