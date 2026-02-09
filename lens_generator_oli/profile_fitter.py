import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image import LensImage

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage



"""
Want to be able to fit a light profile to an image - that will allow us to fit a mass profile as well.
"""

class ProfileFitter:

    def __init__(self, image: str | np.ndarray) -> None:
        if isinstance(image, str):
            image = np.load(image)

    