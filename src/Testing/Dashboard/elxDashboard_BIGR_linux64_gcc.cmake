# Elastix Example Dashboard Script
#
# Copy this example script and edit as necessary for your client.
# See elxDashboardCommon.cmake for more instructions.

# Client maintainer: someone@users.sourceforge.net
set(CTEST_SITE "bigrcluster.erasmusmc")
set(CTEST_BUILD_NAME "Linux-64bit-gcc4.1.2")
#set(CTEST_BUILD_FLAGS "-j2") # parallel build for makefiles
set(CTEST_BUILD_CONFIGURATION Release)
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_DASHBOARD_ROOT "/home/sklein/tk/elastix/nightlybuild" )

# default: automatically determined
#set(CTEST_UPDATE_COMMAND /path/to/svn)

# default: Nightly
#set(dashboard_model Experimental)
#set(dashboard_model Continuous)

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

// Compile all elastix components; todo: automate this
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


# Load the common dashboard script.
include(${CTEST_SCRIPT_DIRECTORY}/elxDashboardCommon.cmake)

