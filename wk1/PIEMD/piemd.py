import numpy as np
from base_classes import Deflector, Source

class PIEMD(Deflector):

    def __init__(self, theta_E: float, e=1.0, s=0.0, phi=0.0) -> None:
        """
        Initialise a pseudo-isothermal elliptical mas distribution (PIEMD) deflector class.
        
        Parameters
        ----------
        theta_E : float
            Einstein radius used to scale the size of the deflector.

        e : float = 1.0
            ellipticity parameter determines how elliptical the resulting convergence map is. 
            A value in (0, 1] where 1 is circular. 
        
        s : float = 0.0
            core parameter. This parameter softens the singularity at the centre of the distribution.

        phi : float = 0.0
            Principal axis angle. This paramater determines the angle of the principal axes of the ellipse to the x and y axes.
            Measured in radians counter-clockwise.
        """

        super().__init__()
        self.theta_E = theta_E

        if e > 1 or e <= 0:
            raise ValueError(f"Ellipticity must be in the range (0, 1], not: e={e}")
        
        self.e = e
        self.s = s
        self.phi = phi

    def get_convergence(self, source: Source):
        src = source.array
        nx, ny, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        x = np.arange(0, nx)
        y = np.arange(0, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy

        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        Re = np.maximum(
            np.sqrt(self.e**2 * (self.s**2 + Rxp**2) + Ryp**2),
            1e-12
        )

        return self.theta_E / (2 * Re)

    def get_potential(self, source: Source):
        src = source.array
        nx, ny, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        x = np.arange(0, nx)
        y = np.arange(0, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy

        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        Re = np.maximum(
            np.sqrt(self.e**2 * (self.s**2 + Rxp**2) + Ryp**2),
            1e-12
        )

        Rf = np.sqrt((1-self.e**2) * Rxp**2 + (Re + self.s)**2) 

        eps = 1e-12
        if abs(1 - self.e) < 1e-6:
            rc = np.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_x = self.theta_E * Rxp / (rc + self.s + eps)
            alpha_y = self.theta_E * Ryp / (rc + self.s + eps)
        
        else:
            a = np.sqrt(1 - self.e**2)

            denom_x = Re + self.s
            denom_y = Re + self.e**2 * self.s

            denom_x = np.maximum(denom_x, eps)
            denom_y = np.maximum(denom_y, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y

            # clip to valid domain for arctanh
            u_y = np.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_x = self.theta_E * (1 / a) * np.arctan(u_x)
            alpha_y = self.theta_E * (1 / a) * np.arctanh(u_y)

        psi =  Rxp * alpha_x + Ryp * alpha_y - self.s * np.log(Rf)

        return self.theta_E * psi


    def get_image(self, source: Source) -> np.ndarray:
        src = source.array
        nx, ny, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        x = np.arange(0, nx)
        y = np.arange(0, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy
        
        # rotate
        Rxp =  np.cos(self.phi) * Rx + np.sin(self.phi) * Ry
        Ryp = -np.sin(self.phi) * Rx + np.cos(self.phi) * Ry

        eps = 1e-12

        Re = np.sqrt(self.e**2 * (self.s**2 + Rxp**2) + Ryp**2)
        Re = np.maximum(Re, eps)

        if abs(1 - self.e) < 1e-6:
            rc = np.sqrt(Rxp**2 + Ryp**2 + self.s**2)
            alpha_x = self.theta_E * Rxp / (rc + self.s + eps)
            alpha_y = self.theta_E * Ryp / (rc + self.s + eps)
        
        else:
            a = np.sqrt(1 - self.e**2)

            denom_x = Re + self.s
            denom_y = Re + self.e**2 * self.s

            denom_x = np.maximum(denom_x, eps)
            denom_y = np.maximum(denom_y, eps)

            u_x = a * Rxp / denom_x
            u_y = a * Ryp / denom_y

            # clip to valid domain for arctanh
            u_y = np.clip(u_y, -1 + 1e-12, 1 - 1e-12)

            alpha_x = self.theta_E * (1 / a) * np.arctan(u_x)
            alpha_y = self.theta_E * (1 / a) * np.arctanh(u_y)


        beta_x = np.rint(X - alpha_x).astype(int)
        beta_y = np.rint(Y - alpha_y).astype(int)

        beta_x = np.clip(beta_x, 0, nx - 1)
        beta_y = np.clip(beta_y, 0, ny - 1)

        image = src[beta_y, beta_x]
        return image
