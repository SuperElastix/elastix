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
# Setup: Windows 7, Visual Studio 9 2008 Win64, Release mode
# Release mode, ITK 4.x (git)
# PC: LKEB (Denis AMD GPU)

# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.PC_AMD" )
set( CTEST_BUILD_NAME "Win7-64bit-VS2008-Release-perf" )
#set( CTEST_BUILD_FLAGS "-j6" ) # parallel build for makefiles
set( CTEST_TEST_ARGS PARALLEL_LEVEL 3 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Visual Studio 9 2008 Win64" )
set( CTEST_DASHBOARD_ROOT "C:/work/elastix/nightly" )
set( CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/bin_release_perf )
set( dashboard_url "https://svn.bigr.nl/elastix/branches/performance_ITK4" )

# Specify the kind of dashboard to submit
# default: Nightly
SET( dashboard_model Nightly )
IF( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  SET( dashboard_model Experimental )
ELSEIF( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  SET( dashboard_model Continuous )
ENDIF()

# Dashboard settings
SET( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=C:/wpackages/ITK-git/build64VC90

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=OFF
ELASTIX_USE_EIGEN:BOOL=OFF
ELASTIX_USE_OPENCL:BOOL=ON
ELASTIX_USE_MEVISDICOMTIFF:BOOL=OFF
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float

// Eigen and OpenCL
OPENCL_INCLUDE_DIRS:PATH=C:/Program Files (x86)/AMD APP/include
OPENCL_LIBRARIES:FILEPATH=C:/Program Files (x86)/AMD APP/lib/x86_64/OpenCL.lib
OPENCL_USE_PLATFORM_AMD:BOOL=ON

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

