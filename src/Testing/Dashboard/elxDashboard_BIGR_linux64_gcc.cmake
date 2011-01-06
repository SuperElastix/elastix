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
# gcc 4.1.2 20061115 (prerelease) (SUSE Linux),
# Release mode, ITK 3.20.0
# PC: linux cluster at BIGR (SK).

# Client maintainer: s.klein@erasmusmc.nl
set( CTEST_SITE "BIGR.cluster" )
set( CTEST_BUILD_NAME "Linux-64bit-gcc4.1.2" )
#set( CTEST_BUILD_FLAGS "-j2" ) # parallel build for makefiles
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Unix Makefiles" )
set( CTEST_DASHBOARD_ROOT "/home/sklein/tk/mydash/${CTEST_SCRIPT_NAME}" )

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

#set(dashboard_do_memcheck 1)
#set(dashboard_do_coverage 1)

SET( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=/home/sklein/tk/itk/3.20/release

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=OFF
ELASTIX_USE_MEVISDICOMTIFF:BOOL=ON
ELASTIX_IMAGE_DIMENSION:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

