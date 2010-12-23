#
# This script runs a dashboard
#
# Setup: Windows 7, Visual Studio 9 2008 Win64, Release mode, ITK 3.20.0
# PC: LKEB, MS personal computer


# Where to find the src directory, and where to build the binaries
SET( CTEST_SOURCE_DIRECTORY "D:/toolkits/elastix/nightly/elastix_src/src" )
SET( CTEST_BINARY_DIRECTORY "D:/toolkits/elastix/nightly/elastix_bin" )

# Where to find CMake on my system
SET( CTEST_CMAKE_COMMAND
  "C:/Program Files (x86)/CMake 2.8/bin/cmake.exe" )

# Specify the kind of dashboard to submit
SET( MODEL Nightly )
IF( ${CTEST_SCRIPT_ARG} MATCHES Experimental )
  SET( MODEL Experimental )
ELSEIF( ${CTEST_SCRIPT_ARG} MATCHES Continuous )
  SET( MODEL Continuous )
ENDIF()
SET( CTEST_COMMAND
  "C:/Program Files (x86)/CMake 2.8/bin/ctest.exe -D ${MODEL}" )
# "C:/Program Files (x86)/CMake 2.8/bin/ctest.exe -D Experimental" )
# "C:/Program Files (x86)/CMake 2.8/bin/ctest.exe -D Continuous" )
# "C:/Program Files (x86)/CMake 2.8/bin/ctest.exe -D Nightly" )

# Specify svn update command
SET( CTEST_SVN_CHECKOUT
  "${CTEST_SVN_COMMAND} co --username elastixguest --password elastixguest https://svn.bigr.nl/elastix/trunkpublic/src \"${CTEST_SOURCE_DIRECTORY}\"" )

# The following does not seem to work:
# find_program( CTEST_SVN_COMMAND NAMES svn )
#SET( CTEST_SVN_COMMAND "C:/cygwin/bin/svn.exe" )
#set( CTEST_UPDATE_COMMAND "${CTEST_SVN_COMMAND}" )
#SET( $ENV{LC_MESSAGES}    "en_EN" )
# should ctest wipe the binary tree before running
SET( CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE )

#SET( CTEST_BUILD_NAME "win7-64-vs2008" )
#set( CTEST_CMAKE_GENERATOR "Visual Studio 9 2008 Win64" )

set( CTEST_BUILD_CONFIGURATION "Release" )
#set(WITH_KWSTYLE FALSE)
#set(WITH_MEMCHECK FALSE)
#set(WITH_COVERAGE FALSE)
#set(WITH_DOCUMENTATION FALSE)

# Usage of the initial cache seems to do the trick:
# Configure the dashboard.
set( CTEST_INITIAL_CACHE "
// Specify build, generator
BUILDNAME:STRING=win7-64-vs2008
CMAKE_GENERATOR:INTERNAL=Visual Studio 9 2008 Win64
CMAKE_CONFIGURATION_TYPES:STRING=Release

SVNCOMMAND:FILEPATH=C:/cygwin/bin/svn.exe

// Which ITK to use
ITK_DIR:PATH=D:/toolkits/ITK/3.20.0/bin

// Some elastix settings, defining the configuration
ELASTIX_BUILD_TESTING:BOOL=ON
ELASTIX_ENABLE_PACKAGER:BOOL=ON
ELASTIX_USE_CUDA:BOOL=ON
ELASTIX_IMAGE_DIMENSION:STRING=2;3;4
ELASTIX_IMAGE_2D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_3D_PIXELTYPES:STRING=float
ELASTIX_IMAGE_4D_PIXELTYPES:STRING=short

// Compile all elastix components
USE_AffineDTITransformElastix:BOOL=ON
USE_BSplineInterpolatorFloat:BOOL=ON
USE_BSplineResampleInterpolatorFloat:BOOL=ON
USE_BSplineTransformWithDiffusion:BOOL=ON
USE_ConjugateGradientFRPR:BOOL=ON
USE_FixedShrinkingPyramid:BOOL=ON
USE_LinearInterpolator:BOOL=ON
USE_LinearResampleInterpolator:BOOL=ON
USE_MovingShrinkingPyramid:BOOL=ON
USE_MutualInformationHistogramMetric:BOOL=ON
USE_NearestNeighborInterpolator:BOOL=ON
USE_NearestNeighborResampleInterpolator:BOOL=ON
USE_RSGDEachParameterApart:BOOL=ON
USE_ViolaWellsMutualInformationMetric:BOOL=ON
")


