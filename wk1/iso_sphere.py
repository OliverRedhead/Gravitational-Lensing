import numpy as np


class Source:
    def __init__(self, size: int, radius: float):
        self.size = size
        self.plane = self._make_disk(radius)

    def _make_disk(self, radius):
        y, x = np.indices((self.size, self.size))
        cx = cy = (self.size - 1) / 2

        r2 = (x - cx)**2 + (y - cy)**2
        return (r2 <= radius**2).astype(float)
    


class Deflector:

    def __init__(self, size, theta_E) -> None:
        self.size = size
        self.theta_E = theta_E
        self.cx = self.cx = self.cy = (size - 1) / 2

    def get_image(self, source):
        image = np.zeros((self.size, self.size))

        for y in range(self.size):
            for x in range(self.size):

                im_val = source.plane[y,x]
                if(im_val == 0):
                    continue

                rx = x - self.cx
                ry = y - self.cy
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

