import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

from base_classes import Source, Deflector, Plane, SIS, PIEMD


import jax
import jax.numpy as jnp
from jax import lax


class DoubleSource:

    def __init__(self, 
                 lens: PIEMD,
                 source1: Plane, 
                 source2: Plane, 
                 theta_E1: float, 
                 theta_E2: float
                 ) -> None:
        
        self.lens = lens
        self.source1 = source1
        self.source2 = source2
        
        self.theta_E1 = theta_E1
        self.theta_E2 = theta_E2

    def get_image(self):
        image = np.zeros_like(self.source2.source.array)
        
        self.lens.theta_E = self.theta_E1
        image += self.lens.lens_source(self.source1.source)

        self.lens.theta_E = self.theta_E2
        image += self.lens.lens_source(self.source2.source)

        return image


        
