import numpy as np
from base_classes import Source, Deflector

class PowerLawDeflector(Deflector):

    def __init__(self, theta_E: float, n: float) -> None:
        super().__init__()
        self.theta_E = theta_E
        self.n = n
        self.kappa_0 = theta_E**(n-1) * (3-n) / 2

    def get_convergence(self, source: Source):
        nx, ny, *_ = source.array.shape
        cx = (nx-1)/2
        cy = (ny-1)/2

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x,y)

        Rx = X - cx
        Ry = Y - cy
        R = np.sqrt(Rx**2 + Ry**2)

        return self.kappa_0 * R**(1-self.n)

    
    def get_image(self, source: Source):
        src = source.array
        nx, ny, *_ = src.shape
    
        cx = (nx - 1)/2
        cy = (ny - 1)/2

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy
        R = np.sqrt(Rx**2 + Ry**2)
        R[R==0] = 1e-6

        alpha_x = Rx * R**(1-self.n) * 2 * self.kappa_0 / (3 - self.n)
        alpha_y = Ry * R**(1-self.n) * 2 * self.kappa_0 / (3 - self.n)

        beta_x = np.rint(X - alpha_x).astype(int)
        beta_y = np.rint(Y - alpha_y).astype(int)

        beta_x = np.clip(beta_x, 0, nx - 1)
        beta_y = np.clip(beta_y, 0, ny - 1)

        image = src[beta_y, beta_x]

        return image





