import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp
from jax import lax

from base_classes import Deflector, Source


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

    def __init__(self, theta_E: float, *, q=1.0, s=0.0, phi=0.0, gamma_1=0.0, gamma_2=0.0) -> None:
        """
        Initialise a pseudo-isothermal elliptical mas distribution (PIEMD) deflector class.
        
        Parameters
        ----------
        theta_E : float
            Einstein radius used to scale the size of the deflector.

        e : float = 1.0
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
            External shear parameter in x=y x=-y direction
        """

        super().__init__()
        self.theta_E = theta_E

        if q > 1 or q <= 0:
            raise ValueError(f"Axis ratio must be in the range (0, 1], not: q={q}")
        
        self.q = q
        self.s = s
        self.phi = phi

        self.g1 = gamma_1
        self.g2 = gamma_2

    def get_convergence(self, source: Source):
        src = source.array
        ny, nx, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

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

    def get_potential(self, source: Source):
        src = source.array
        ny, nx, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

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
    

    def get_image(self, source: Source) -> np.ndarray:
        src = source.array
        ny, nx, *_ = src.shape

        nc = 1
        if src.ndim == 2:
            ny, nx = src.shape
        if src.ndim == 3:
            ny, nx, nc = src.shape # type:ignore

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

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

        return image
        

    def map(self, source: Source, r: tuple[float, float]):
        """
        given a coordinate on image plane (x,y), returns corresponding point on source plane.
        """
        x, y = r

        src = source.array
        ny, nx, *_ = src.shape

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

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
    
    def map_jax(self, source, r):
        
        x,y = r[0], r[1]
        ny, nx, *_ = source.array.shape

        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

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

    def get_magnification(self, source: Source):

        def map_only_r(r):
            return self.map_jax(source, r)

        jac_map = jax.jacfwd(map_only_r)

        # Optional but highly recommended:
        jac_map = jax.jit(jac_map)

        X,Y = source.get_indices()
        R = np.stack((X, Y), axis=-1).astype(np.float32)   # shape (nx, ny, 2)
        R_flat = R.reshape(-1, 2)

        map_per_point = lambda r: self.map_jax(source, r)
        A_flat = jax.vmap(jax.jacfwd(map_per_point))(R_flat)
        A = A_flat.reshape(X.shape[0], X.shape[1], 2, 2)

        a = A[...,0,0]*A[...,1,1] - A[...,1,0]*A[...,0,1] # inverse magnification
        return a
    
    def get_critical_curves(self, source: Source):
        
        a = self.get_magnification(source)
        cs = plt.contour(a, levels=[0])

        segments = cs.allsegs
        
        # TODO turn this into something useable

        pass






    
        

        