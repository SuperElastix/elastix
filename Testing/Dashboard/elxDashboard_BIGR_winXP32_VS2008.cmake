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
# Setup: Windows XP, 32bit, Visual Studio 9 2008, Release mode, ITK 4.1.0
# PC: BIGR, SK personal computer.

# Client maintainer: s.klein@erasmusmc.nl
set( CTEST_SITE "BIGR.PCStefan" )
set( CTEST_BUILD_NAME "WinXP-32bit-VS2008" )
#set( CTEST_BUILD_FLAGS "-j2" ) # parallel build for makefiles
set( CTEST_TEST_ARGS PARALLEL_LEVEL 2 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Release )
set( CTEST_CMAKE_GENERATOR "Visual Studio 9 2008" )

# default: automatically determined
#set(CTEST_UPDATE_COMMAND /path/to/svn)
# this does not work, because quotes are put around it later:
#set(CTEST_UPDATE_COMMAND "C:/Program Files/Subversion/bin/svn.exe --config-dir d:/dox/rest/confignoext")
# this gives strange xml parse errors:
#set(CTEST_UPDATE_COMMAND "D:/scripts/svn.bat")
# this works:
set(CTEST_UPDATE_COMMAND "C:/Program Files/Subversion/bin/svn.exe")

# Specify the kind of dashboard to submit
# default: Nightly
SET( dashboard_model Nightly )
IF( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  SET( dashboard_model Experimental )
ELSEIF( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  SET( dashboard_model Continuous )
ENDIF()

# name of output directory
set( CTEST_DASHBOARD_ROOT "D:/tk/mydash/${CTEST_SCRIPT_NAME}.${dashboard_model}" )

#set(dashboard_do_memcheck 1)
#set(dashboard_do_coverage 1)

SET( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=D:/tk/itk/4.5.0/bin

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=OFF
ELASTIX_USE_MEVISDICOMTIFF:BOOL=ON
ELASTIX_IMAGE_DIMENSIONS:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Compile all elastix components;
USE_ALL_COMPONENTS:BOOL=ON
")


# Load the common dashboard script.
include( ${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake )

