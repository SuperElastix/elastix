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
# Setup: MacOSX 64bit, 10.6.5
# gcc 4.2.1
# Release mode, ITK 3.20.0
# PC: LKEB (MS), MacMini PC of Patrick de Koning

# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.MacMini" )
set( CTEST_BUILD_NAME "MacOSX-64bit-gcc4.2.1" )
set( CTEST_BUILD_FLAGS "-j2" ) # parallel build for makefiles
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Unix Makefiles" )
set( CTEST_DASHBOARD_ROOT "/elastix-nightly/build/" )

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

#set(dashboard_do_memcheck 1)
#set(dashboard_do_coverage 1)

set( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=/elastix-nightly/itk/git/bin_release

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=OFF
ELASTIX_USE_MEVISDICOMTIFF:BOOL=OFF
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

