#---------------------------------------------------------------------
Project( elxTesting )

#---------------------------------------------------------------------
# Add macro if mevisdicomtiff should be used
IF( ELASTIX_USE_MEVISDICOMTIFF )
  ADD_DEFINITIONS( -D_ELASTIX_USE_MEVISDICOMTIFF )
  SET( mevisdcmtifflib mevisdcmtiff )
ELSE()
  SET( mevisdcmtifflib )
ENDIF()

#---------------------------------------------------------------------
# Macro that simplifies the addition of tests
#
# Usage:
# ADD_ELX_TEST( <name_of_test> )
#
# We assume here that the name of the source which includes the main is:
#   "itk"name_of_test".cxx".
#

MACRO( ADD_ELX_TEST name )

  # Define some helper variables.
  SET( ELXTEST_EXECUTABLE_NAME itk${name} )
  SET( ELXTEST_SOURCE_NAME itk${name}.cxx )
  SET( ELXTEST_TEST_NAME ${name} )

  # Create the test executable.
  ADD_EXECUTABLE( ${ELXTEST_EXECUTABLE_NAME} ${ELXTEST_SOURCE_NAME} )

  # Link against other libraries.
  TARGET_LINK_LIBRARIES( ${ELXTEST_EXECUTABLE_NAME}
    ITKCommon ITKBasicFilters elxCommon ${mevisdcmtifflib} )

  # Group in IDE's like Visual Studio
  SET_PROPERTY( TARGET ${ELXTEST_EXECUTABLE_NAME} PROPERTY FOLDER "tests" )

  # Add the test.
  ADD_TEST( ${ELXTEST_TEST_NAME}
    ${EXECUTABLE_OUTPUT_PATH}/${ELXTEST_EXECUTABLE_NAME} ${ARGN} )

ENDMACRO( ADD_ELX_TEST )

#---------------------------------------------------------------------

ADD_ELX_TEST( AdvancedBSplineDeformableTransformTest
  ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt )
ADD_ELX_TEST( BSplineDerivativeKernelFunctionTest )
ADD_ELX_TEST( BSplineSODerivativeKernelFunctionTest )
ADD_ELX_TEST( BSplineInterpolationWeightFunctionTest )
ADD_ELX_TEST( BSplineInterpolationDerivativeWeightFunctionTest )
ADD_ELX_TEST( BSplineInterpolationSODerivativeWeightFunctionTest )
ADD_ELX_TEST( MevisDicomTiffImageIOTest )
ADD_ELX_TEST( ThinPlateSplineTransformPerformanceTest
  ${elastix_SOURCE_DIR}/Testing/parameters_TPSTransformTest.txt
  ${elastix_BINARY_DIR}/Testing )
ADD_ELX_TEST( ThinPlateSplineTransformTest
  ${elastix_SOURCE_DIR}/Testing/parameters_TPSTransformTest.txt )
ADD_ELX_TEST( TimerTest )


