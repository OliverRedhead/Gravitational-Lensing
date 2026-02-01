import numpy as np

from piemd import PIEMD, Deflector, Source


class DoubleLens:

    def __init__(self, lens: PIEMD, source1: Deflector, source2: Source) -> None:
        self.source1 = source1
        self.source2 = source2
        self.lens = lens

    

    
