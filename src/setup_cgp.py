# System imports
from distutils.core import *
from distutils      import sysconfig
import numpy

## Third-party modules - we depend on numpy for everything
#import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# cgp extension module
_cgp = Extension("_cgp",
            ["cgp.i","cgp.c","gpr.c","covfunc.c","utility.c"],
            include_dirs = [numpy_include,'.'],
            )

# NumyTypemapTests setup
setup(  name        = "C-version of Gaussian Processes Regression",
        description = "Implement training and predicting algorithm of GPR in the book GPML.",
        author      = "Wei Gong",
        version     = "1.0",
        ext_modules = [_cgp]
        )
