# Elastix Dashboard Script
#
# This script runs a dashboard
# Usage:
#   ctest -S <nameofthisscript> -V
#   OR
#   ctest -S <nameofthisscript>,Model -V
#
# It has 1 optional argument: the build model.
# The build model should be one of {Experimental, Continuous, Nightly}
# and defaults to Nightly.
# NOTE that Model should directly follow the comma: no space allowed!
#
# Setup: Windows 7, Visual Studio 9 2008 Win64, Release mode, ITK 3.20.0
# PC: LKEB, MS personal computer


# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.PCMarius" )
set( CTEST_BUILD_NAME "Win7-64bit-VS2008" )
set( CTEST_TEST_ARGS PARALLEL_LEVEL 3 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Visual Studio 9 2008 Win64" )
set( CTEST_DASHBOARD_ROOT "D:/toolkits/elastix/nightly" )

# default: automatically determined
#set(CTEST_UPDATE_COMMAND /path/to/svn)

# Specify the kind of dashboard to submit
# default: Nightly
SET( dashboard_model Nightly )
IF( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  SET( dashboard_model Experimental )
ELSEIF( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  SET( dashboard_model Continuous )
ENDIF()

# Note:
# ITK4 can currently not be used in combination with CUDA
# The commandline for nvcc is too long
# A bug is outstanding: http://www.itk.org/Bug/view.php?id=12198
SET( dashboard_cache "
// Which ITK to use:
ITK_DIR:PATH=D:/toolkits/ITK/git/binVS2008

// Some elastix settings, defining the configuration:
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
//ELASTIX_USE_CUDA:BOOL=ON
ELASTIX_USE_MEVISDICOMTIFF:BOOL=OFF
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Compile all elastix components:
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

