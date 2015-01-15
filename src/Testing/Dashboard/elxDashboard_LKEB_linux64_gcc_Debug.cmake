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
# Debug mode, ITK 4.x (git), code coverage by gcov
# PC: LKEB (MS), goliath

# Client maintainer: m.staring@lumc.nl
set( CTEST_SITE "LKEB.goliath" )
set( CTEST_BUILD_NAME "Linux-64bit-gcc4.8.2-Debug" )
set( CTEST_BUILD_FLAGS "-j6" ) # parallel build for makefiles
set( CTEST_TEST_ARGS PARALLEL_LEVEL 6 ) # parallel testing
set( CTEST_BUILD_CONFIGURATION Debug )
set( CTEST_CMAKE_GENERATOR "Unix Makefiles" )
set( CTEST_DASHBOARD_ROOT "/home/marius/nightly-builds/elastix" )
set( CTEST_BINARY_DIRECTORY ${CTEST_DASHBOARD_ROOT}/bin_debug )

# Specify the kind of dashboard to submit
# default: Nightly
set( dashboard_model Nightly )
if( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  set( dashboard_model Experimental )
elseif( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  set( dashboard_model Continuous )
endif()

# This machine performs code coverage analysis and dynamic memory checking.
set( dashboard_do_coverage ON )
set( dashboard_do_memcheck ON )

# Valgrind options
#set( CTEST_MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --quiet --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=100 --verbose --demangle=yes --gen-suppressions=all" )
#set( CTEST_MEMORYCHECK_SUPPRESSIONS_FILE ${CTEST_SOURCE_DIRECTORY}/CMake/InsightValgrind.supp )

# Dashboard settings
set( dashboard_cache "
// Which ITK to use
ITK_DIR:PATH=/usr/local/toolkits/ITK/git/bin_debug

// Coverage settings: -fprofile-arcs -ftest-coverage
CMAKE_CXX_FLAGS_DEBUG:STRING=-g -O0 -fprofile-arcs -ftest-coverage
CMAKE_C_FLAGS_DEBUG:STRING=-g -O0 -fprofile-arcs -ftest-coverage
CMAKE_EXE_LINKER_FLAGS_DEBUG:STRING=-g -O0 -fprofile-arcs -ftest-coverage

// Memory check setting for valgrind:
//MEMORYCHECK_COMMAND_OPTIONS:STRING=${CTEST_MEMORYCHECK_COMMAND_OPTIONS}
//MEMORYCHECK_SUPPRESSIONS_FILE:FILEPATH=${CTEST_MEMORYCHECK_SUPPRESSIONS_FILE}

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=ON
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

