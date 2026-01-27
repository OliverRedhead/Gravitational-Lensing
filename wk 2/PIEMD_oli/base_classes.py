import numpy as np
import matplotlib.pyplot as plt

class Source:

    def __init__(self, array : np.ndarray = np.array([])) -> None:
        self.array = array
        self.colour = array.shape[-1] == 3

    def get_source(self) -> np.ndarray:
        return self.array

    def pad(self, pad):
        arr = self.array

        if arr.ndim == 2:
            # Grayscale image (H, W)
            H, W = arr.shape
            padded = np.zeros((H + 2*pad, W + 2*pad))
            padded[pad:pad+H, pad:pad+W] = arr

        elif arr.ndim == 3:
            # RGB image (H, W, 3)
            H, W, C = arr.shape # type:ignore
            assert C == 3

            padded = np.zeros((H + 2*pad, W + 2*pad, 3))
            padded[pad:pad+H, pad:pad+W, :] = arr
            padded = padded / np.max(padded)

        else:
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB)")

        self.array = padded
        return self.array.shape
    
    def get_indices(self):
        ny, nx, *_ = self.array.shape

        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)

        X, Y = np.meshgrid(x,y)

        return X,Y
    
    def plot_source(self):
        plt.imshow(self.array, origin='lower')
        plt.show()

    
class DiskSource(Source):

    """
    A class which will represent the source in a lensing system.
    
    Specifically represents a uniform disk of light as an `n x n` numpy array.
    """

    def __init__(self, size: int, radius: float):
        """
        intialises a DiskSource instance.

        Parameters
        ----------
        size : int
            the size of the square source array.

        radius : float
            the radius of the uniform disk
        """
        self.size = size
        self.radius = radius
        super().__init__(self._make_disk())

    def _make_disk(self):
        """
        initialiser method which fills the source array with a disk of 
        radius `radius` centered at center of the array
        """
        y, x = np.indices((self.size, self.size))
        cx = cy = (self.size - 1) / 2

        r2 = (x - cx)**2 + (y - cy)**2
        return (r2 <= self.radius**2).astype(float)



class Deflector:

    def __init__(self, source: Source = Source()) -> None:
        self.source = source
        self.ny, self.nx, *_ = self.source.array.shape

    def get_image(self) -> np.ndarray:
        return self.source.array
    
    def get_convergence(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def get_potential(self) -> np.ndarray:
        return np.zeros_like(self.source.array)
    
    def set_source(self, new_source: Source):
        self.source = new_source
        self.ny, self.nx = self.source.array.shape

    def align_source(self, cx: int = 0, cy: int = 0, source=False ):
        """
        sets the pixel given (cx,cy) as the center of the lens
        """
        self.cx = cx
        self.cy = cy
        return
    
    def pad_source(self, pad):
        self.ny, self.nx, *_ = self.source.pad(pad) 
        self.cx = (self.nx-1) / 2.0
        self.cy = (self.ny-1) / 2.0