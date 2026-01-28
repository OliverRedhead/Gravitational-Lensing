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

    def __init__(self, source: Source, *, theta_E=None, q=1.0, s=0.0, phi=0.0, gamma_1=0.0, gamma_2=0.0, lens_colour=[1.0,1.0,1.0]) -> None:
        """
        Initialise a pseudo-isothermal elliptical mas distribution (PIEMD) deflector class.
        
        Parameters
        ----------
        theta_E : float
            Einstein radius used to scale the size of the deflector.

        q : float = 1.0
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
            External shear parameter in x=y, x=-y direction
        """

        super().__init__(source)
        
        if isinstance(theta_E, float) or isinstance(theta_E, int):
            self.theta_E = theta_E
        else:
            # Default Einstein radius: 0.3 times the larger image dimension
            self.theta_E = 0.3 * max(self.source.array.shape)

        if q > 1 or q <= 0:
            raise ValueError(f"Axis ratio must be in the range (0, 1], not: q={q}")
        
        self.q = q
        self.s = s
        self.phi = phi

        self.g1 = gamma_1
        self.g2 = gamma_2

        ny, nx, *_ = self.source.array.shape

        self.cx = (nx-1) / 2.0
        self.cy = (ny-1) / 2.0

        self.colour = source.colour
        self.lens_colour = np.array(lens_colour)

    def generate_image(self) -> np.ndarray:
        src = self.source.array
        nx, ny, nc = self.nx, self.ny, 1

        if src.ndim == 2:
            ny, nx = src.shape
        if src.ndim == 3: # if colour
            ny, nx, nc = src.shape # type:ignore

        cx = self.cx
        cy = self.cy


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

        if self.colour:
            Imax = np.max(image)
            if(Imax > 0):
                image = image / np.max(image)
            Imax = np.clip(image, 0.0, 1.0) 

        return image

    def generate_lens(self, n=2):
        """
        Generate a 2D elliptical Gaussian representing a lens galaxy.

        Parameters
        ----------
        shape : tuple of int
            (ny, nx) size of the output image.
        einstein_radius : float
            Characteristic radius (in pixels) of the lens (similar to sigma for Gaussian).
        axis_ratio : float
            Minor-to-major axis ratio q = b/a (0 < q <= 1).
        I0 : float
            Peak intensity.

        Returns
        -------
        np.ndarray
            2D image of the elliptical Gaussian lens.
        """
        nx, ny = self.nx, self.ny
        cx, cy = self.cx, self.cy

        Y, X = np.indices((ny, nx))

        # Center coordinates
        dx = X - cx
        dy = Y - cy

        # Rotate coordinates
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        X_rot = cos_phi * dx + sin_phi * dy
        Y_rot = -sin_phi * dx + cos_phi * dy

        # Elliptical radius
        r2 = (self.q * X_rot)**2 + Y_rot**2
        r2 = r2**(n/2)  # only if you want a generalized Gaussian with power n

        # Elliptical Gaussian
        img = np.exp(-0.5 * r2 / self.theta_E**2)

        return img

    def generate_noise(self):
        colour = self.source.array.shape[-1] 
        if (colour == 3):
            noise = np.random.normal(0, 0.1, size=(self.nx, self.ny, colour)) # TODO stdev can be set automatically somehow
        else:
            noise = np.random.normal(0, 0.1, size=(self.nx, self.ny)) 
        return noise

    def generate_hubble_image(self):
        # get image plane
        img = self.generate_image()
        img = img / np.max(img)
        
        # Superimpose lens galaxy
        img += self.generate_lens()
        
        # clip
        img_lens = np.clip(img, 0, 1.0) # TODO check how this looks without this
        
        # zoom to hubble resolution
        img_zoomed = self.hubble_resolution(img_lens)

        # add noise characteristic of hubble 
        img_noise = self.hubble_noise(img_zoomed, gain=4, dark_level='high')

        # blur with Hubble psf
        img_blurred = self.hubble_blur(img_noise)

        return img_blurred


    """
    two mapping methods - one for normal mapping and one for jax.
    They should both work the exact same way.
    """

    def map(self, r):
        """
        given a coordinate on image plane (x,y), returns corresponding point on source plane.
        """
        
        x, y = r

        cx = self.cx
        cy = self.cy

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
    
    def map_jax(self, r):
        
        x,y = r[0], r[1]
        nx, ny, *_ = self.nx, self.ny

        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

        cx = self.cx
        cy = self.cy

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

    """
    getter methods for testing
    """

    def get_convergence(self):
        nx, ny = self.nx, self.ny

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        cx = self.cx
        cy = self.cy

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

    def get_potential(self):
        nx, ny = self.nx, self.ny

        cx = (nx - 1) / 2
        cy = (ny - 1) / 2

        cx = self.cx
        cy = self.cy

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

    def get_magnification(self):

        def map_only_r(r):
            return self.map_jax(r)

        jac_map = jax.jacfwd(map_only_r)

        # Optional but highly recommended:
        jac_map = jax.jit(jac_map)

        X,Y = self.source.get_indices()
        R = np.stack((X, Y), axis=-1).astype(np.float32)   # shape (nx, ny, 2)
        R_flat = R.reshape(-1, 2)

        map_per_point = lambda r: self.map_jax(r)
        A_flat = jax.vmap(jax.jacfwd(map_per_point))(R_flat)
        A = A_flat.reshape(X.shape[0], X.shape[1], 2, 2)

        a = A[...,0,0]*A[...,1,1] - A[...,1,0]*A[...,0,1] # inverse magnification
        return a
    
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
        mapped_curves = [np.array([self.map(point) for point in polygon]) for polygon in curves]
        return mapped_curves

    """
    These methods help me view things
    """

    def plot_source(self, *, critical_curves=False, caustics=False, 
                    convergence=False, potential=False, image=True, noise=False):

        # initialise noise array
        noise_arr = np.zeros_like(self.source.array)
        if noise:
            noise_arr = self.generate_noise()

        # colour
        if self.source.array.shape[-1] == 3:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.source.array

            image_arr += noise_arr
            image_arr = np.clip(image_arr, 0.0, 1.0)

            plt.imshow(image_arr, origin='lower')

        else:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.source.array

            image_arr += noise_arr

            cbar = plt.imshow(image_arr, cmap='inferno', origin='lower')
            plt.colorbar(cbar)

        if convergence:
            plt.contour(self.get_convergence())

        if potential:
            plt.contour(self.get_potential())

        if critical_curves:
            for curve in self.get_critical_curves():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='red')

        if caustics:
            for curve in self.get_caustics():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='orange')

        plt.show()

    def plot_image(self, *, critical_curves=False, caustics=False, 
                convergence=False, potential=False, image=True, lens=True, noise=False, I_0=2.0):

        # initialise noise array
        noise_arr = np.zeros_like(self.source.array)
        if noise:
            noise_arr = self.generate_noise()

        # colour
        if self.source.array.shape[-1] == 3:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.get_image()

            if lens:
                image_arr += self.generate_lens()

            image_arr += noise_arr
            image_arr = np.clip(image_arr, 0.0, 1.0)


            plt.imshow(image_arr, origin='lower')

        else:
            image_arr = np.zeros_like(self.source.array, dtype=float)

            if image:
                image_arr += self.get_image()

            if lens:
                image_arr += self.generate_lens()

            image_arr += noise_arr

            cbar = plt.imshow(image_arr, cmap='inferno', origin='lower')
            plt.colorbar(cbar)

        if convergence:
            plt.contour(self.get_convergence())

        if potential:
            plt.contour(self.get_potential())

        if critical_curves:
            for curve in self.get_critical_curves():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='red')

        if caustics:
            for curve in self.get_caustics():
                plt.plot(curve[:, 0], curve[:, 1], '-', c='orange')

        plt.show()



    
        

        