from base_classes import Source, Deflector
import numpy as np

class SIS(Deflector):

    """
    Defelctor class for Singular Isothermal Sphere
    """

    def __init__(self, theta_E: float):
        super().__init__()
        self.theta_e = theta_E

    def convergence(self, source: Source):
        
        nx, ny, *_ = source.array.shape
    
        cx = (nx-1)/2
        cy = (ny-1)/2

        x = np.linspace(0,nx,1) 
        y = np.linspace(0,ny,1)
        X , Y = np.meshgrid(x,y, indexing="xy")

        Rx = X - cx
        Ry = Y - cy

        theta = np.sqrt(Rx**2+Ry**2)

        return theta

    
    """
    def potential(self, source: Source):


    def image(self, source: Source):
    """
    
