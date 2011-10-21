#---------------------------------------------------------------------
Project( elxTesting )

# Set some directories
set( TestDataDir ${elastix_SOURCE_DIR}/Testing/Data)
set( TestBaselineDir ${elastix_SOURCE_DIR}/Testing/Baselines)
set( TestOutputDir ${elastix_BINARY_DIR}/Testing )


#---------------------------------------------------------------------
# Python is used by some tests
find_program( python_executable python )


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
# elx_add_test( <name_of_test> )
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
# elx_add_run_test( <name_of_test>
#                   IMAGE <name_of_baseline> | METRICVALUE "string"
#                   <elastix_commandline_arguments> )
#

# Python script used for comparing elastix output with baseline strings
set( pythonfinalmetric ${elastix_SOURCE_DIR}/Testing/elx_compare_finalmetricvalue.py )
set( pythonchecksum ${elastix_SOURCE_DIR}/Testing/elx_compare_checksum.py )

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
macro( elx_add_run_test testbasename howtocompare baseline )
  # Create output directory
  set( testname elastix_run_${testbasename} )
  set( output_dir ${TestOutputDir}/${testname} )
  file( MAKE_DIRECTORY ${output_dir} )

  # Number of -p's to deduce result.?.mhd
  list_count( "${ARGN}" "-p" countp )
  math( EXPR countp "${countp} - 1" )

  # Run elastix, but only in release mode
  # Set maximum test length to 10 minutes
  add_test( NAME ${testname}_OUTPUT
    CONFIGURATIONS Release
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/elastix
    ${ARGN}
    -out ${output_dir} )
  set_tests_properties( ${testname}_OUTPUT
    PROPERTIES TIMEOUT 600 )
  # Compare elastix output image with baseline image
  if( ${howtocompare} STREQUAL "IMAGE" )
    add_test( NAME ${testname}_COMPARE
      CONFIGURATIONS Release
      COMMAND elxImageCompare
      -base ${baseline}
      -test ${output_dir}/result.${countp}.mhd )
    set_tests_properties( ${testname}_COMPARE
      PROPERTIES DEPENDS  ${testname}_OUTPUT )
  # Compare elastix output from IterationInfo with a baseline string
  elseif( ${howtocompare} STREQUAL "METRICVALUE" )
    # Only run when python was found
    if( python_executable )
      add_test( NAME ${testname}_COMPARE
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonfinalmetric}
        -b ${baseline} -d ${output_dir} )
      set_tests_properties( ${testname}_COMPARE
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()
  # Compare elastix output using checksum
  elseif( ${howtocompare} STREQUAL "CHECKSUM" )
    # Only run when python was found
    if( python_executable )
      add_test( NAME ${testname}_COMPARE
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonchecksum}
        -b ${baseline} -d ${output_dir} -v )
      set_tests_properties( ${testname}_COMPARE
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()

  endif()
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
  "IMAGE" ${TestBaselineDir}/example.mha
  -f ${ELASTIX_DOX_DIR}/exampleinput/fixed.mhd
  -m ${ELASTIX_DOX_DIR}/exampleinput/moving.mhd
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_Rigid.txt
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_BSpline.txt )

# Run 3D registration with a 'common' parameter file:
elx_add_run_test( 3DCT_lung.example
  "IMAGE" ${TestBaselineDir}/3DCT_lung.mha
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001.txt )

# Test some transforms
elx_add_run_test( 3DCT_lung.NC.translation.ASGD.001
  "METRICVALUE" "199 -0.852717 19.336725 27.739195 0.005591 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.translation.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.euler.ASGD.001
  "METRICVALUE" "199 -0.871208 37.674389 12.100341 0.005748 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.euler.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.affine.ASGD.001metric
  "METRICVALUE" "499 -0.917434 147.732214 6.247953 0.005693 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.001.txt )

# this checksum is made on win32, visual2008
elx_add_run_test( 3DCT_lung.NC.affine.ASGD.001checksum
  "CHECKSUM" 2212427187
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.001.txt )

# Test some metrics
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001
  "METRICVALUE" "99 7096.689847 32.307007 0.019832 855.319746 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.ASGD.001
  "METRICVALUE" "499 -1.136692 187.234245 603.377080 0.004836 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lun.NMI.bspline.ASGD.001
  "METRICVALUE" "99 -1.342389 61.264050 4558.392221 0.002384 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NMI.bspline.ASGD.001.txt )

# Test some combo metrics
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.002
  "METRICVALUE" "99 -0.989393 -0.989431 0.037435 45.501739 5506.194266 0.001166 0.104286 0.001171 x x x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fp ${TestDataDir}/3DCT_lung_baseline.txt
  -mp ${TestDataDir}/3DCT_lung_followup.txt
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.003
  "METRICVALUE" "49 -0.981680 -0.981777 0.000971 10.673041 23783.231320 0.000699 0.000015 0.000699 x x x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.003.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.004
  "METRICVALUE" "49 -0.969652 -0.979208 0.095563 16.496939 21033.767226 0.000800 0.000652 0.000798 0.0001288746 0.0602022919 0.0112364764 x x x 0.0001707896 0.0003848968 0.0002337033"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.004.txt )

# Test some optimizers
elx_add_run_test( 3DCT_lung.NC.bspline.SGD.001
  "METRICVALUE" "99 -0.933783 56.557637 0.000768 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.QN.001
  "METRICVALUE" "99 99 0 -0.992011 1.000000 0.000871 3.651948 0.001700 x x x x x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.QN.001.txt )

# Test some samplers
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001
  "METRICVALUE" "99 -0.936908 56.557637 0.004449 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.002
  "METRICVALUE" "99 -1.035576 56.557637 0.001980 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.002.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.003
  "METRICVALUE" "2 -0.826970 304.784793 0.001133 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.003.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.004
  "METRICVALUE" "99 -0.292445 56.557637 0.002204 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fMask ${TestDataDir}/3DCT_lung_baseline_mask.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.004.txt )

# Test some interpolators
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.002
  "METRICVALUE" "99 -0.985803 30.804875 9703.960481 0.001238 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.003
  "METRICVALUE" "99 -0.979751 24.438035 8486.512890 0.000785 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.003.txt )

