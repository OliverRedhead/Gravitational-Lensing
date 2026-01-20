import numpy as np
from base_classes import Source, Deflector

class EllipticalDeflector(Deflector):

    def __init__(self, theta_E : float, gamma_1: float, gamma_2: float) -> None:
        super().__init__()
        self.theta_E = theta_E
        self.g1 = gamma_1
        self.g2 = gamma_2


    def get_image(self, source: Source) -> np.ndarray:
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

        alpha_x = self.theta_E * Rx/R + 2 * self.g1 * Rx + 2 * self.g2 * Ry
        alpha_y = self.theta_E * Ry/R + 2 * self.g2 * Rx - 2 * self.g1 * Ry 

        beta_x = np.rint(X - alpha_x).astype(int)
        beta_y = np.rint(Y - alpha_y).astype(int)

        beta_x = np.clip(beta_x, 0, nx - 1)
        beta_y = np.clip(beta_y, 0, ny - 1)

        image = src[beta_y, beta_x]

        return image

