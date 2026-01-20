import numpy as np
from base_classes import Source, Deflector


class EinsteinDeflector(Deflector):

    """
    A class to calculate the deflection of an isothermal sphere.
    """

    def __init__(self, theta_E: float) -> None:
        super().__init__()
        self.theta_E = theta_E

    def get_convergence(self, source: Source):
        """
        A getter method for visualising the surface density of the lens.

        Parameters
        ----------
        source : Source
            A source to set the dimensions of the output convergence field
        """
        
        nx, ny, *_ = source.array.shape
        cx = (nx-1)/2
        cy = (nx-1)/2
        
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x,y, indexing='xy')

        Rx = X - cx
        Ry = Y - cy
        theta = np.sqrt(Rx**2 + Ry**2)
        theta[theta == 0] = 1e-6

        kappa = self.theta_E / (2 * theta)

        return kappa
    
    def get_potential(self, source: Source):
        """
        A getter method for visualising the lensing potential of the lens.

        Parameters
        ----------
        source : Source
            A source to set the dimensions of the output potential field
        """
        
        nx, ny, *_ = source.array.shape
        cx = (nx-1)/2
        cy = (nx-1)/2
        
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x,y, indexing='xy')

        Rx = X - cx
        Ry = Y - cy
        theta = np.sqrt(Rx**2 + Ry**2)
        theta[theta == 0] = 1e-6

        kappa = self.theta_E * theta

        return kappa
    

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
        nx, ny , *others = src.shape

        cx = (nx - 1)/2
        cy = (ny - 1)/2

        for y in range(ny):
            for x in range(nx):

                rx = x - cx
                ry = y - cy
                r = np.sqrt(rx**2 + ry**2)
                if(r == 0):
                    r = 1e-6
                
                alpha_x = self.theta_E * rx / r
                alpha_y = self.theta_E * ry / r

                beta_x = round(x - alpha_x)
                beta_y = round(y - alpha_y)

                image[y,x] += src[beta_y, beta_x]

        self.image = image
        return image

