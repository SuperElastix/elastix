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
# Setup: Windows 10, Visual Studio 14 2015 Win64, Release mode, ITK 4.11.1
# PC: LKEB, MS personal computer


# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.PCMarius" )
set( CTEST_BUILD_NAME "Win10-64bit-VS2015" )
set( CTEST_TEST_ARGS PARALLEL_LEVEL 3 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Visual Studio 14 2015 Win64" )
set( CTEST_DASHBOARD_ROOT "D:/toolkits/elastix/nightly" )
set( CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/bin_VS2015 )

# default: automatically determined
#set(CTEST_UPDATE_COMMAND /path/to/svn)

# Specify the kind of dashboard to submit
# default: Nightly
set( dashboard_model Nightly )
if( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  set( dashboard_model Experimental )
elseif( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  set( dashboard_model Continuous )
endif()

set( dashboard_cache "
// Which ITK to use:
ITK_DIR:PATH=D:/toolkits/ITK/latest_release/bin_VS2015

// Some elastix settings, defining the configuration:
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_EIGEN:BOOL=OFF
ELASTIX_USE_OPENCL:BOOL=OFF
ELASTIX_USE_MEVISDICOMTIFF:BOOL=OFF
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Eigen and OpenCL
OPENCL_INCLUDE_DIRS:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.0/include
OPENCL_LIBRARIES:FILEPATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v6.0/lib/x64/OpenCL.lib
OPENCL_USE_NVIDIA_SDK:BOOL=ON

// Compile all elastix components:
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

