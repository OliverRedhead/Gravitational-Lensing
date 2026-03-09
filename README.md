# Gravitational-Lensing
This repository will hold the work I do over the Dennison scholarship on simulating gravitational lensing over the summer.

All the methods you want are in ``euclid_generator/euclid_generator.py``. This contains the EuclidGenerator class which does all the hard work for me. I will make sure to properly comment the methods and things in there, but the important ones are 
``EuclidGenerator.sample_kwargs(self, key, image_index, lens_rotation=0)`` and
``get_model(self, mass_kwargs, light_kwargs, eta, unconvolved=False, source=None)``

The ``testing()`` method should provide an example of how I use the class, there are also some examples in the `testing.ipynb` file. Note that the class expects the workng directory to be `Gravitational-Lensing`.

You will need to download the .fits files for the foreground images (there are too many to hold on github), these should go in the `euclid_generator/lrg_in` folder. Everything else should be ready to run. 

# TODO
This is not a complete list, but this is all I can come up with off the top of my head.
- Still not happy with how blobby the sources are
- Lensed sources need noise before being painted on.
- einstien radius of source 2 is not generated properly. There are some more notes in the EuclidGenerator.__sample_s1_kwargs() method.
  

Again, thank you for your time and effort over this project, it has been a very rewarding experience =). Have fun!