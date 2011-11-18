#---------------------------------------------------------------------
Project( elxTesting )

# Set some directories
set( TestDataDir ${elastix_SOURCE_DIR}/Testing/Data)
set( TestBaselineDir ${elastix_SOURCE_DIR}/Testing/Baselines)
set( TestOutputDir ${elastix_BINARY_DIR}/Testing )


#---------------------------------------------------------------------
# Python is used by some tests
find_program( python_executable python
  HINTS "C:/Program Files/Python2.7" )
if( NOT python_executable )
  message( WARNING "WARNING: python not found! Some test results can not be evaluated." )
endif()

#---------------------------------------------------------------------
# Add macro if mevisdicomtiff should be used
if( ELASTIX_USE_MEVISDICOMTIFF )
  add_definitions( -D_ELASTIX_USE_MEVISDICOMTIFF )
  set( mevisdcmtifflib mevisdcmtiff )
else()
  set( mevisdcmtifflib )
endif()


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
      -test ${output_dir}/result.${countp}.mhd
      -t 1.0 )
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
  "CHECKSUM" 2711524819
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.translation.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.euler.ASGD.001
  "CHECKSUM" 3164583274
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.euler.ASGD.001.txt )

# temporarily keep this test:
elx_add_run_test( 3DCT_lung.NC.affine.ASGD.001metric
  "METRICVALUE" "499 -0.917434 147.732214 6.247953 0.005693 x"#
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.affine.ASGD.001checksum
  "CHECKSUM" 2212427187
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.001.txt )

# Test some metrics
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001
  "CHECKSUM" 2209466441
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.ASGD.001
  "CHECKSUM" 2234356863
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NMI.bspline.ASGD.001
  "CHECKSUM" 3201757062
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NMI.bspline.ASGD.001.txt )

# Test some combo metrics
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.002
  "CHECKSUM" 1374321245
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fp ${TestDataDir}/3DCT_lung_baseline.txt
  -mp ${TestDataDir}/3DCT_lung_followup.txt
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.003
  "CHECKSUM" 392325787
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.003.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.004
  "CHECKSUM" 3464268329
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.004.txt )

# Test some optimizers
elx_add_run_test( 3DCT_lung.NC.bspline.SGD.001
  "CHECKSUM" 3274775574
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.QN.001
  "CHECKSUM" 869048287
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.QN.001.txt )

# Test some samplers
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001
  "CHECKSUM" 947853335
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.002
  "CHECKSUM" 3495426412
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.002.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.003
  "CHECKSUM" 1158326297
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.003.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.004
  "CHECKSUM" 2710903580
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fMask ${TestDataDir}/3DCT_lung_baseline_mask.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.004.txt )

# Test some interpolators
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.002
  "CHECKSUM" 1740515246
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.003
  "CHECKSUM" 2043650983
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.003.txt )



### TEMPORARY TESTING TO FIND THE PLATFORM INCONSISTENCIES
# Checksums defined on windows 64 bit machine LKEB PC Marius
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001a
  "CHECKSUM" 792071748
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001a.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001b
  "CHECKSUM" 2618589300
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001b.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001c
  "CHECKSUM" 3105983252
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001c.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001d
  "CHECKSUM" 2549300057
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001d.txt )

