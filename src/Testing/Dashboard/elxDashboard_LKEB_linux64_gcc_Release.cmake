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
# Setup: Linux 64bit, Ubuntu 14.04 LTS, 3.13.0-24-generic
# gcc 4.8.2
# Release mode, ITK 4.x (git)
# PC: LKEB (MS), goliath

# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.goliath" )
set( CTEST_BUILD_NAME "Linux-64bit-gcc4.8.2-Release" )
set( CTEST_BUILD_FLAGS "-j6" ) # parallel build for makefiles
set( CTEST_TEST_ARGS PARALLEL_LEVEL 6 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Unix Makefiles" )
set( CTEST_DASHBOARD_ROOT "/home/marius/nightly-builds/elastix" )
set( CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/bin_release )

# Specify the kind of dashboard to submit
# default: Nightly
set( dashboard_model Nightly )
if( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  set( dashboard_model Experimental )
elseif( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  set( dashboard_model Continuous )
endif()

# Dashboard settings
set( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=/srv/lkeb-goliath/toolkits/ITK/git/bin_release

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_EIGEN:BOOL=ON
ELASTIX_USE_OPENCL:BOOL=ON
ELASTIX_USE_MEVISDICOMTIFF:BOOL=OFF
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Eigen and OpenCL
OPENCL_INCLUDE_DIRS:PATH=/usr/local/cuda/include
OPENCL_LIBRARIES:FILEPATH=/usr/lib/libOpenCL.so
OPENCL_USE_PLATFORM_NVIDIA:BOOL=ON
EIGEN3_INCLUDE_DIR:PATH=/home/marius/toolkits/eigen/eigen-3.2.1

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

