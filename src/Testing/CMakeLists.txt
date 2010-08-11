#---------------------------------------------------------------------
Project( elxTesting )

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
    ITKCommon ITKBasicFilters )

  # Add the test.
  ADD_TEST( ${ELXTEST_TEST_NAME} ${ELXTEST_EXECUTABLE_NAME} ${ARGN} )

ENDMACRO( ADD_ELX_TEST )

#---------------------------------------------------------------------

ADD_ELX_TEST( BSplineDerivativeKernelFunctionTest )
ADD_ELX_TEST( BSplineSODerivativeKernelFunctionTest )
ADD_ELX_TEST( BSplineInterpolationWeightFunctionTest )
ADD_ELX_TEST( BSplineInterpolationDerivativeWeightFunctionTest )
ADD_ELX_TEST( BSplineInterpolationSODerivativeWeightFunctionTest )
ADD_ELX_TEST( AdvancedBSplineDeformableTransformTest ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt)
