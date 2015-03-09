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
# Setup: linux 64bit
# gcc 4.4.6 (Redhat Linux),
# Release mode, ITK 4.7.1
# PC: linux cluster at BIGR (SK).

# Client maintainer: s.klein@erasmusmc.nl
set( CTEST_SITE "BIGR.cluster" )
set( CTEST_BUILD_NAME "Linux-64bit-gcc4.4.6" )
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Unix Makefiles" )

# default: automatically determined
#set(CTEST_UPDATE_COMMAND /path/to/svn)

# Specify the kind of dashboard to submit
# default: Nightly
set( dashboard_model Nightly )
set( CTEST_BUILD_FLAGS "-j3" ) # parallel build for makefiles
set( CTEST_TEST_ARGS PARALLEL_LEVEL 3 ) # parallel testing

if( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  set( dashboard_model Experimental )
elseif( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  set( dashboard_model Continuous )
endif()

# Output directory
set( CTEST_DASHBOARD_ROOT "/cm/shared/apps/elastix/nightly" )

# Use 'release' instead of 'bin', for consistency with other
# elastix versions on the cluster2
set( CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/release )

#set(dashboard_do_memcheck 1)
#set(dashboard_do_coverage 1)

set( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=/cm/shared/apps/itk/4.7.1/release

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=OFF
ELASTIX_USE_MEVISDICOMTIFF:BOOL=ON
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=short;float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Install dir
CMAKE_INSTALL_PREFIX:PATH=/cm/shared/apps/elastix/nightly/install

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

