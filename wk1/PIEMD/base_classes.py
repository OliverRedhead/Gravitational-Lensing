import numpy as np

class Source:

    def __init__(self, array : np.ndarray) -> None:
        self.array = array

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

    def __init__(self) -> None:
        pass

