import astropy.constants as const
import numpy as np

class PointMass:
    """
    This class represents a point mass for the purposes of simulating gravitational lensing.
    
    We assume a "weak lens" where newtonian gravitational potential per mass Phi = -GM/r << c^2 => Phi/c^2 << 1
    and use the born approximation to approximate the grravitational potential along the deflected path
    by the gravitational potential along the un-deflected path (valid for small deflection angles)
    """

    def __init__(self, M) -> None:
        self.M = M # mass

    def get_deflection(self, xi, arcseconds=False):
        """
        Given an impact parameter (xi) returns the deflection angle (alpha : float).
        Returns alpha in radians if arcseconds=False and arcseconds if arcseconds=True
        """
        alpha_rad = 4 * const.G.value * self.M / (const.c.value**2 * xi) # type: ignore
        alpha_arcsec = alpha_rad * (180 * 3600 / np.pi)
        if(arcseconds):
            return alpha_arcsec
        return alpha_rad

		def func(self, a):
					return a