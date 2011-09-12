#---------------------------------------------------------------------
Project( elxTesting )

# Set some directories
set( TestDataDir ${elastix_SOURCE_DIR}/Testing/Data)
set( TestBaselineDir ${elastix_SOURCE_DIR}/Testing/Baselines)
set( TestOutputDir ${elastix_BINARY_DIR}/Testing )

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

macro( elx_add_test name )
  # Define some helper variables.
  set( ELXTEST_EXECUTABLE_NAME itk${name} )
  set( ELXTEST_SOURCE_NAME itk${name}.cxx )
  set( ELXTEST_TEST_NAME ${name} )

  # Create the test executable.
  add_executable( ${ELXTEST_EXECUTABLE_NAME} ${ELXTEST_SOURCE_NAME} )

  # Link against other libraries.
  target_link_libraries( ${ELXTEST_EXECUTABLE_NAME}
    ITKCommon ITKBasicFilters elxCommon ${mevisdcmtifflib} )

  # Group in IDE's like Visual Studio
  set_property( TARGET ${ELXTEST_EXECUTABLE_NAME} PROPERTY FOLDER "tests" )

  # Add the test.
  add_test( NAME ${ELXTEST_TEST_NAME}
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/${ELXTEST_EXECUTABLE_NAME} ${ARGN} )
endmacro()

#---------------------------------------------------------------------
# Macro that simplifies the addition of tests that run elastix
#
# Usage:
# ADD_ELX_TEST( <name_of_test> )
#
# We assume here that the name of the source which includes the main is:
#   "itk"name_of_test".cxx".
#

# Helper macro
macro( list_count listvar value count )
  set( tmplist )
  foreach( tmpvalue ${listvar} )
    if( ${tmpvalue} STREQUAL ${value} )
      list( APPEND tmplist . )
    endif()
  endforeach()
  list( LENGTH tmplist tmpcount )
  set( ${count} ${tmpcount} )
endmacro()

# Main elastix_run add_test macro
macro( elx_add_run_test testbasename baseline )
  # Create output directory
  set( testname elastix_run_${testbasename} )
  set( output_dir ${TestOutputDir}/${testname} )
  file( MAKE_DIRECTORY ${output_dir} )

  # Number of -p's to deduce result.?.mhd
  list_count( "${ARGN}" "-p" countp )
  math( EXPR countp "${countp} - 1" )

  # Run elastix and compare output with baseline
  add_test( NAME ${testname}_OUTPUT
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/elastix
    ${ARGN}
    -out ${output_dir} )
  add_test( NAME ${testname}_COMPARE
    COMMAND elxImageCompare
    -base ${baseline}
    -test ${output_dir}/result.${countp}.mhd )
  set_tests_properties( ${testname}_COMPARE
    PROPERTIES DEPENDS  ${testname}_OUTPUT )
endmacro()

#---------------------------------------------------------------------

# Create imagecompare.
add_executable( elxImageCompare elxImageCompare.cxx itkCommandLineArgumentParser.cxx )
target_link_libraries( elxImageCompare ${ITK_LIBRARIES} )
set_property( TARGET elxImageCompare PROPERTY FOLDER "tests" )

#---------------------------------------------------------------------
# Add tests

elx_add_test( AdvancedBSplineDeformableTransformTest
  ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt )
elx_add_test( BSplineDerivativeKernelFunctionTest )
elx_add_test( BSplineSODerivativeKernelFunctionTest )
elx_add_test( BSplineInterpolationWeightFunctionTest )
elx_add_test( BSplineInterpolationDerivativeWeightFunctionTest )
elx_add_test( BSplineInterpolationSODerivativeWeightFunctionTest )
elx_add_test( MevisDicomTiffImageIOTest )
elx_add_test( ThinPlateSplineTransformPerformanceTest
  ${elastix_SOURCE_DIR}/Testing/parameters_TPSTransformTest.txt
  ${elastix_BINARY_DIR}/Testing )
elx_add_test( ThinPlateSplineTransformTest
  ${elastix_SOURCE_DIR}/Testing/parameters_TPSTransformTest.txt )
elx_add_test( TimerTest )

#---------------------------------------------------------------------
# Add tests that run elastix

# Run example
elx_add_run_test( example
  ${TestBaselineDir}/example.mha
  -f ${ELASTIX_DOX_DIR}/exampleinput/fixed.mhd
  -m ${ELASTIX_DOX_DIR}/exampleinput/moving.mhd
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_Rigid.txt
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_BSpline.txt )

elx_add_run_test( 3DCT_lung
  ${TestBaselineDir}/3DCT_lung.mha
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.A.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.A.txt )

