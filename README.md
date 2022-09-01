<a href="https://elastix.lumc.nl/">
  <img src="https://github.com/SuperElastix/elastix/blob/main/dox/art/elastix_logo_full_small.bmp" alt="elastix logo" title="elastix" align="right" height="80" />
</a>

# elastix image registration toolbox #

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/SuperElastix/elastix/raw/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/itk-elastix.svg)](https://pypi.python.org/pypi/itk-elastix)
[![GitHub Actions](https://github.com/SuperElastix/elastix/workflows/Elastix/badge.svg)](https://github.com/SuperElastix/elastix/actions)
[![Model Zoo](https://img.shields.io/badge/open-Model%20Zoo-blue.svg)](https://elastix.lumc.nl/modelzoo/)
[![Docker](https://img.shields.io/badge/open-docker%20image-blueviolet.svg)](https://hub.docker.com/repository/docker/superelastix/elastix)

Welcome to elastix: a toolbox for rigid and nonrigid registration of images.

elastix is open source software, based on the well-known [Insight Segmentation and Registration Toolkit (ITK)](https://itk.org/). The software consists of a collection of algorithms that are commonly used to solve (medical) image registration problems. The modular design of elastix allows the user to quickly configure, test, and compare different registration methods for a specific application. A command-line interface enables automated processing of large numbers of data sets, by means of scripting.
Nowadays elastix is accompanied by [ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix) making it available in Python ([on Pypi](https://pypi.org/project/itk-elastix/)) and by [SimpleElastix](http://simpleelastix.github.io/), making it available in languages like C++, Python, Java, R, Ruby, C# and Lua. A docker image of the latest elastix build is available as well on [dockerhub](https://hub.docker.com/repository/docker/superelastix/elastix). Several plugins exist for those who wish to use the functionality of elastix in a graphical user interface, among others a [napari](https://github.com/SuperElastix/elastix_napari) and a [3Dslicer](https://github.com/lassoan/SlicerElastix) plugin.

## Authors ##

The lead developers of elastix are [Stefan Klein](https://github.com/stefanklein) and [Marius Staring](https://github.com/mstaring). This software was initially developed at the [Image Sciences Institute](http://www.isi.uu.nl), under supervision of [Josien P.W. Pluim](http://www.isi.uu.nl/People/Josien/). Today, [many](https://github.com/SuperElastix/elastix/graphs/contributors) have contributed to elastix.

If you use this software anywhere we would appreciate if you cite the following articles:
- S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, "elastix: a toolbox for intensity based medical image registration," IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010. [download](https://elastix.lumc.nl/marius/publications/2010_j_TMI.php) [doi](http://dx.doi.org/10.1109/TMI.2009.2035616)
- D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification of Alzheimer’s Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014. [download](https://elastix.lumc.nl/marius/publications/2014_j_FNI.php) [doi](http://dx.doi.org/10.3389/fninf.2013.00050)

Specific components of elastix are made by many; The relevant citation can be found [here](https://github.com/SuperElastix/elastix/wiki/How-to-cite-elastix-(components)).

## More information ##

More information, including an extensive manual and model zoo, can be found on the [wiki](https://github.com/SuperElastix/elastix/wiki)

Interactive tutorials are available in [Jupyter notebooks](https://mybinder.org/v2/gh/InsightSoftwareConsortium/ITKElastix/master?urlpath=lab/tree/examples%2FITK_Example01_SimpleRegistration.ipynb).

You can also subscribe to the [mailing list](https://groups.google.com/forum/#!forum/elastix-imageregistration) for questions. Information on contributing to `elastix` can be found [here](CONTRIBUTING.md).
