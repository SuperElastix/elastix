#---------------------------------------------------------------------
project( elxTesting )

# Set some directories
set( TestDataDir ${elastix_SOURCE_DIR}/Testing/Data )
set( TestBaselineDir ${elastix_SOURCE_DIR}/Testing/Baselines )
set( TestOutputDir ${elastix_BINARY_DIR}/Testing )

set( SITE_BASELINE_DIR "" CACHE PATH "path to site-specific baseline data" )
if( SITE_BASELINE_DIR )
  if( EXISTS ${SITE_BASELINE_DIR} )
    set( TestSiteBaselineDir ${SITE_BASELINE_DIR} )
    message( STATUS "Site-specific baseline directory: " ${TestSiteBaselineDir} )
  else()
    message( FATAL_ERROR "Invalid site-specific baseline directory specified: " ${TestSiteBaselineDir} )
  endif()
endif()

#---------------------------------------------------------------------
# configure the baseline files
set( ELASTIX_DATA_DIR ${TestDataDir} )
file( GLOB files ${TestBaselineDir}/*.in )
if( TestSiteBaselineDir )
  file( GLOB sitefiles ${TestSiteBaselineDir}/*.in )
endif()
foreach( tpfile ${files} ${sitefiles} )
  # Get file name without directory
  get_filename_component( tpfileout ${tpfile} NAME )
  # Strip extension .in
  string( REGEX REPLACE "\\.[^.]*$" "" tpfileout ${tpfileout})
  # Replace the initial transform directory with the correct one
  # The resulting file can be found in the binary directory / Testing
  configure_file( ${tpfile} ${elastix_BINARY_DIR}/Testing/${tpfileout} @ONLY )
endforeach()

#---------------------------------------------------------------------
# Python is used by some tests
find_program( python_executable python
  HINTS "C:/Program Files/Python2.7" "C:/Program Files/python3.3" "C:/Program Files/Python35" )
if( NOT python_executable )
  message( WARNING "WARNING: python not found! Some test results can not be evaluated." )
endif()

#---------------------------------------------------------------------
# Add macro to decide if mevisdicomtiff should be tested
if( ELASTIX_USE_MEVISDICOMTIFF )
  add_definitions( -D_ELASTIX_USE_MEVISDICOMTIFF )
endif()
set( mevisdcmtifflib mevisdcmtiff )

#---------------------------------------------------------------------
# Optionally run timing tests
# - tests that are wholly devoted to timing can be added conditionally using cmake symbol ELASTIX_TEST_TIMING
# - tests that have mixed correctness and timing testing can use the C Preprocessor symbol _ELASTIX_TEST_TIMING to
#   conditionally enable the timing testing parts
mark_as_advanced( ELASTIX_TEST_TIMING )
option( ELASTIX_TEST_TIMING "Enable timing tests?  This option enables timing tests among the regular test suite." OFF )
if( ELASTIX_TEST_TIMING )
  add_definitions( -D_ELASTIX_TEST_TIMING )
endif()

#---------------------------------------------------------------------
# Core macro for elx_add_opencl_test() and elx_add_test()
macro( elx_add_test_core executable_name source_name test_name group )
  if( NOT TARGET ${executable_name} )

    # Create the test executable.
    add_executable( ${executable_name} ${source_name}
      itkCommandLineArgumentParser.cxx # some test use the CommandLineArgumentParser
    )

    # Link against other libraries.
    target_link_libraries( ${executable_name}
      param               # some test use the CommandLineArgumentParser
      ${mevisdcmtifflib}  # is empty if not selected in CMake
      ${ITK_LIBRARIES}
    )

    if( ELASTIX_USE_OPENCL )
      include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
      target_link_libraries( ${executable_name} elxOpenCL )
    endif()

    # Group in IDE's like Visual Studio
    set_property( TARGET ${executable_name} PROPERTY FOLDER "tests/${group}" )
  endif()

  # Add the test.
  add_test( NAME ${test_name}
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/${executable_name} ${ARGN} )

  # GPU test should be run in serial, to avoid memory issues.
  string( COMPARE EQUAL ${group} "OpenCL" isEqual )
  if( ${isEqual} )
    set_tests_properties( ${test_name} PROPERTIES RUN_SERIAL true )
  endif()
endmacro()

#---------------------------------------------------------------------
# Macro that simplifies the addition of elastix OpenCL tests
#
# Usage:
# elx_add_opencl_test( <name_of_test> )
#
# We assume here that the name of the source which includes the main is:
#   "itk"name_of_test".cxx".
#

macro( elx_add_opencl_test name number group kernels )
  # Define some helper variables.
  set( ELXTEST_EXECUTABLE_NAME itk${name} )
  set( ELXTEST_SOURCE_NAME itk${name}.cxx itkTestHelper.h itkTestOutputWindow.cxx itkTestOutputWindow.h )
  set( ELXTEST_TEST_NAME ${name}${number} )

  # Count number of kernels provided
  set( all_kernels ${kernels} )
  list( LENGTH all_kernels ELXTEST_OPENCL_NUM_KERNELS )
  if( ${ELXTEST_OPENCL_NUM_KERNELS} GREATER 0 )

    # Kernels source variables
    set( ELXTEST_OPEN_CL_KERNELS )
    set( ELXTEST_OPEN_CL_KERNELS_SRC )

    # If single kernel
    if( ${ELXTEST_OPENCL_NUM_KERNELS} EQUAL 1 )
      # Define OpenCL kernels
      file( GLOB open_cl_kernel_path ${kernels} )
      list( APPEND ELXTEST_OPEN_CL_KERNELS ${open_cl_kernel_path} )

      source_group( "Kernels" FILES ${ELXTEST_OPEN_CL_KERNELS} )
      write_opencl_kernels( "${ELXTEST_OPEN_CL_KERNELS}" ELXTEST_OPEN_CL_KERNELS_SRC "OpenCLTest generated kernels" OFF ${name} )
    # Else if multiple kernels kernel
    elseif( ${ELXTEST_OPENCL_NUM_KERNELS} GREATER 1 )
      # Define OpenCL kernels
      foreach( elem ${kernels} )
        file( GLOB open_cl_kernel_path ${elem} )
        list( APPEND ELXTEST_OPEN_CL_KERNELS ${open_cl_kernel_path} )
      endforeach()

      source_group( "Kernels" FILES ${ELXTEST_OPEN_CL_KERNELS} )
      write_opencl_kernels( "${ELXTEST_OPEN_CL_KERNELS}" ELXTEST_OPEN_CL_KERNELS_SRC "OpenCLTest generated kernels" ON ${name} )
    endif()

    # Append to ELXTEST_SOURCE_NAME
    list( APPEND ELXTEST_SOURCE_NAME ${ELXTEST_OPEN_CL_KERNELS} ${ELXTEST_OPEN_CL_KERNELS_SRC} itk${name}.h )
  endif()

  # Call core macro
  elx_add_test_core( ${ELXTEST_EXECUTABLE_NAME} "${ELXTEST_SOURCE_NAME}" ${ELXTEST_TEST_NAME} ${group} ${ARGN} )
endmacro()

#---------------------------------------------------------------------
# Macro that simplifies the addition of elastix tests
#
# Usage:
# elx_add_test( <name_of_test> )
#
# We assume here that the name of the source which includes the main is:
#   "itk"name_of_test".cxx".
#

macro( elx_add_test name number group )
  # Define some helper variables.
  set( ELXTEST_EXECUTABLE_NAME itk${name} )
  set( ELXTEST_SOURCE_NAME itk${name}.cxx )
  set( ELXTEST_TEST_NAME ${name}${number} )

  # Call core macro
  elx_add_test_core( ${ELXTEST_EXECUTABLE_NAME} "${ELXTEST_SOURCE_NAME}" ${ELXTEST_TEST_NAME} ${group} ${ARGN} )
endmacro()

#---------------------------------------------------------------------
# Macro that simplifies the addition of transformix tests
#
# Usage:
# trx_add_test( <name_of_test> <transformix_commandline_arguments> )
#
# The macro runs transformix with the provided commandline arguments.
#

macro( trx_add_test name )
  # Create output directory
  set( testname transformix_run_${name} )
  set( output_dir ${TestOutputDir}/${testname} )
  file( MAKE_DIRECTORY ${output_dir} )

  # Define some helper variables.
  set( TRXTEST_EXECUTABLE_NAME transformix )
  set( TRXTEST_TEST_NAME ${name} )

  # Group in IDE's like Visual Studio
  set_property( TARGET ${ELXTEST_EXECUTABLE_NAME} PROPERTY FOLDER "tests" )

  # Add the test.
  add_test( NAME ${name} COMMAND ${EXECUTABLE_OUTPUT_PATH}/transformix ${ARGN} -out ${output_dir} )
  set_tests_properties( ${name} PROPERTIES TIMEOUT 10000 )
endmacro()

#---------------------------------------------------------------------
# Macro's that simplifies the addition of tests that run elastix
#
# Usage:
# elx_add_run_test( <name_of_test>
#                   IMAGE <name_of_baseline> | METRICVALUE "string"
#                     | CHECKSUM "string"
#                   <elastix_commandline_arguments> )
#

# Get the baseline checksum values file
#
# Ideally, a registration returns exactly the same result, regardless of
# platform, OS and compiler. We observed however, that this is not the case.
# Therefore, for some tests we accept differences between systems, and only
# test for consistency over time for the same system.
#
# To this end we have defined baseline checksum values for several systems.
# The code below checks what the current system is and finds the corresponding
# baseline file. We currently do this by means of the CTEST_SITE and CTEST_BUILD_NAME.
# Different sites can give different results, so we only test the checksum for
# the sites that are known to us, and those are the sites that submit to the
# nightly dashboard. Note that this means that these checksum comparison test
# only work if run via the dashboard scripts.
#
set( knownChecksumSystem false )
set( specializedTransformParameterSystem false )
if( SITE STREQUAL "LKEB.PCMarius"
    AND BUILDNAME STREQUAL "Win10-64bit-VS2015" )
  set( checksumBaselineFile ${TestBaselineDir}/checksums_LKEB.PCMarius_Win10-64bit-VS2015.txt )
  set( knownChecksumSystem true )
  set( specializedTransformParameterSystem true )
elseif( SITE STREQUAL "LKEB-ELDB91"
    AND BUILDNAME STREQUAL "Linux-64bit-gcc5.4.0-Release" )
  set( checksumBaselineFile ${TestBaselineDir}/checksums_LKEB.ELDB91_Linux-64bit-gcc5.txt )
  set( knownChecksumSystem true )
  set( specializedTransformParameterSystem true )
elseif( SITE STREQUAL "LKEB.MacMini"
    AND BUILDNAME STREQUAL "MacOSX-64bit-gcc4.2.1" )
  set( checksumBaselineFile ${TestBaselineDir}/checksums_LKEB.MacMini_MacOSX-64bit-gcc4.txt )
  set( knownChecksumSystem true )
  set( specializedTransformParameterSystem true )
elseif( SITE STREQUAL "BIGR.PCStefan"
    AND BUILDNAME STREQUAL "WinXP-32bit-VS2008" )
  set( checksumBaselineFile ${TestBaselineDir}/checksums_BIGR.PCStefan_WinXP-32bit-VS2008.txt )
  set( knownChecksumSystem true )
  set( specializedTransformParameterSystem true )
elseif( SITE STREQUAL "BIGR.cluster"
    AND BUILDNAME STREQUAL "Linux-64bit-gcc4.4.6" )
  set( checksumBaselineFile ${TestBaselineDir}/checksums_BIGR.cluster_Linux-64bit-gcc4.txt )
  set( knownChecksumSystem true )
  set( specializedTransformParameterSystem true )
endif()

if( EXISTS ${TestSiteBaselineDir}/checksums.txt )
  set( checksumBaselineFile ${TestSiteBaselineDir}/checksums.txt )
  set( knownChecksumSystem true )
endif()

if( knownChecksumSystem )
  message( STATUS "Checksum baseline file: " ${checksumBaselineFile} )
endif()

# Python script used for comparing elastix output with baseline results
set( pythonfinalmetric ${elastix_SOURCE_DIR}/Testing/elx_compare_finalmetricvalue.py )
set( pythonchecksum   ${elastix_SOURCE_DIR}/Testing/elx_compare_checksum.py )
set( pythonoverlap    ${elastix_SOURCE_DIR}/Testing/elx_compare_overlap.py )
set( pythonlandmarks  ${elastix_SOURCE_DIR}/Testing/elx_compare_landmarks.py )

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
macro( elx_add_run_test testbasename howtocompare )
  # Create output directory
  set( testname elastix_run_${testbasename} )
  set( output_dir ${TestOutputDir}/${testname} )
  file( MAKE_DIRECTORY ${output_dir} )

  # ${ARGN} may be the elastix command line arguments, but also
  # an optional baseline value followed by those arguments.
  set( baseline ${ARGV2} )
  set( elastixargs ${ARGN} )
  string( SUBSTRING ${baseline} 0 1 firstelement )
  string( COMPARE EQUAL ${firstelement} "-" firstelementisdash )
  if( NOT ${firstelementisdash} )
    list( REMOVE_AT elastixargs 0 )
  endif()

  # Number of -p's to deduce result.?.mhd
  list_count( "${ARGN}" "-p" countp )
  math( EXPR countp "${countp} - 1" )

  # Run elastix, but only in release mode
  # Set maximum test length to 10 minutes
  add_test( NAME ${testname}_OUTPUT
    CONFIGURATIONS Release
    COMMAND ${EXECUTABLE_OUTPUT_PATH}/elastix
    ${elastixargs} -out ${output_dir} )
  set_tests_properties( ${testname}_OUTPUT
    PROPERTIES TIMEOUT 600 )

  # To compare against a baseline TransformParameters file
  # Baselines have been generated on LKEB, PCMarius, Win10, 64 bit, VS2015
  set( baselineTP ${TestOutputDir}/TransformParameters_${testbasename}.txt )
  set( testTP ${output_dir}/TransformParameters.${countp}.txt )

  # Temporary hack to get some failing tests working
  # These three tests give different results on different platforms
  # They all seem to be using ASGD and B-splines and (N)MI
  if( specializedTransformParameterSystem )
    if( ( SITE STREQUAL "LKEB-ELDB91" ) OR (SITE STREQUAL "LKEB.MacMini") OR (SITE STREQUAL "BIGR.cluster") OR (SITE STREQUAL "BIGR.PCStefan") )
      if( ${testbasename} STREQUAL "3DCT_lung.MI.bspline.ASGD.001"
        OR ${testbasename} STREQUAL "3DCT_lung.NMI.bspline.ASGD.001" )
        set( baselineTP ${TestOutputDir}/TransformParameters_${SITE}_${testbasename}.txt )
      endif()
    endif()
  endif()

  # Thread tests should use the original TransformParameters file as a baseline
  string( FIND ${testbasename} "Threads" found )
  if( NOT found EQUAL -1 )
    string( REGEX REPLACE "(-Threads[0-9]+)" "" baselineTP ${baselineTP} )
  endif()

  # Check which tests have to be run
  string( REGEX MATCHALL "[a-zA-Z]+;|[a-zA-Z]+$" compareaslist "${howtocompare}" )
  list( FIND compareaslist "IMAGE"       compare_image )
  list( FIND compareaslist "METRICVALUE" compare_metricvalue )
  list( FIND compareaslist "PARAMETERS"  compare_tp )
  list( FIND compareaslist "CHECKSUM"    compare_checksum )
  list( FIND compareaslist "OVERLAP"     compare_overlap )
  list( FIND compareaslist "LANDMARKS"   compare_landmarks )

  # Compare elastix output image with baseline image
  if( NOT ${compare_image} EQUAL -1 )
    add_test( NAME ${testname}_COMPARE_IM
      CONFIGURATIONS Release
      COMMAND elxImageCompare
      -base ${baseline}
      -test ${output_dir}/result.${countp}.mhd
      -t 5 -a 50 )
    set_tests_properties( ${testname}_COMPARE_IM
      PROPERTIES DEPENDS  ${testname}_OUTPUT )
  endif()

  # Compare elastix output from IterationInfo with a baseline string
  if( NOT ${compare_metricvalue} EQUAL -1 )
    # Only run when python was found
    if( python_executable )
      # Add comparison test, checking the final metric value and more
      add_test( NAME ${testname}_COMPARE_METRIC
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonfinalmetric}
        -b ${baseline} -d ${output_dir} )
      set_tests_properties( ${testname}_COMPARE_METRIC
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()
  endif()

  # Compare elastix output using checksum
  if( NOT ${compare_checksum} EQUAL -1 )
    # Only run when python was found and if this is a known system
    if( python_executable AND knownChecksumSystem )
      # Add comparison test, checking the registration checksum
      add_test( NAME ${testname}_COMPARE_CHECKSUM
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonchecksum}
        -f ${checksumBaselineFile} -d ${output_dir} -v )
      set_tests_properties( ${testname}_COMPARE_CHECKSUM
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()
  endif()

  # Compare elastix output using segmentation overlap
  if( NOT ${compare_overlap} EQUAL -1 )
    # Only run when python was found
    if( python_executable )
      add_test( NAME ${testname}_COMPARE_OVERLAP
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonoverlap}
        -b ${baselineTP} -d ${output_dir} -m ${TestDataDir}/3DCT_lung_followup_mask.mha
        -p $<TARGET_FILE:elxComputeOverlap> )
      set_tests_properties( ${testname}_COMPARE_OVERLAP
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()
  endif()

  # Compare elastix output using landmark distance
  if( NOT ${compare_landmarks} EQUAL -1 )
    # Only run when python was found
    if( python_executable )
      add_test( NAME ${testname}_COMPARE_LANDMARKS
        CONFIGURATIONS Release
        COMMAND ${python_executable} ${pythonlandmarks}
        -b ${baselineTP} -d ${output_dir} -f ${TestDataDir}/3DCT_lung_baseline_landmarks.txt -t 1.0
        -p $<TARGET_FILE:transformix> )
      set_tests_properties( ${testname}_COMPARE_LANDMARKS
        PROPERTIES DEPENDS  ${testname}_OUTPUT )
    endif()
  endif()

  # Compare elastix output using transform parameters comparison
  if( NOT ${compare_tp} EQUAL -1 )
    add_test( NAME ${testname}_COMPARE_TP
      CONFIGURATIONS Release
      COMMAND elxTransformParametersCompare
      -base ${baselineTP}
      -test ${testTP}
      -a 1e-3 )
    set_tests_properties( ${testname}_COMPARE_TP
      PROPERTIES DEPENDS  ${testname}_OUTPUT )
  endif()
endmacro()

#---------------------------------------------------------------------

# Create elxComputeOverlap
add_executable( elxComputeOverlap elxComputeOverlap.cxx itkCommandLineArgumentParser.cxx )
target_link_libraries( elxComputeOverlap ${ITK_LIBRARIES} )
set_property( TARGET elxComputeOverlap PROPERTY FOLDER "tests/Executable" )

# Create elxImageCompare
add_executable( elxImageCompare elxImageCompare.cxx itkCommandLineArgumentParser.cxx )
target_link_libraries( elxImageCompare ${ITK_LIBRARIES} )
set_property( TARGET elxImageCompare PROPERTY FOLDER "tests/Executable" )

# Create elxTransformParametersCompare
add_executable( elxTransformParametersCompare elxTransformParametersCompare.cxx itkCommandLineArgumentParser.cxx )
target_link_libraries( elxTransformParametersCompare param ${ITK_LIBRARIES} )
set_property( TARGET elxTransformParametersCompare PROPERTY FOLDER "tests/Executable" )

# Create elxInvertTransform
add_executable( elxInvertTransform elxInvertTransform.cxx itkCommandLineArgumentParser.cxx )
target_link_libraries( elxInvertTransform param ${ITK_LIBRARIES} )
set_property( TARGET elxInvertTransform PROPERTY FOLDER "tests/Executable" )

#---------------------------------------------------------------------
# Add tests

# Add a test for inverting an affine transform
# Add a test for comparing the inverse against the ground truth
add_test( NAME InvertTransformTest_OUTPUT
  COMMAND elxInvertTransform
  -tp  ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -out ${TestOutputDir}/TransformParameters_3DCT_lung.affine.inverse.txt
  -m   ${TestDataDir}/3DCT_lung_followup.mha )
add_test( NAME InvertTransformTest_COMPARE_TP
  COMMAND elxTransformParametersCompare
  -base ${TestBaselineDir}/TransformParameters_3DCT_lung.affine.inverse.txt
  -test ${TestOutputDir}/TransformParameters_3DCT_lung.affine.inverse.txt
  -a 1e-3 )
set_tests_properties( InvertTransformTest_COMPARE_TP
  PROPERTIES DEPENDS InvertTransformTest_OUTPUT )

# Add tests that run specific registration components
elx_add_test( AdvancedBSplineDeformableTransformTest "" "Common"
  ${TestDataDir}/parameters_AdvancedBSplineDeformableTransformTest.txt )
elx_add_test( AdvancedRecursiveBSplineTransformTest "" "Common"
  ${TestDataDir}/parameters_AdvancedBSplineDeformableTransformTestSml.txt )
elx_add_test( AdvancedLinearInterpolatorTest "" "Common" )
elx_add_test( BSplineDerivativeKernelFunctionTest "" "Common" )
elx_add_test( BSplineSODerivativeKernelFunctionTest "" "Common" )
elx_add_test( BSplineInterpolationWeightFunctionTest "" "Common" )
elx_add_test( BSplineInterpolationDerivativeWeightFunctionTest "" "Common" )
elx_add_test( BSplineInterpolationSODerivativeWeightFunctionTest "" "Common" )
elx_add_test( CompareCompositeTransformsTest "" "Common" )
elx_add_test( MevisDicomTiffImageIOTest "" "Common" )
elx_add_test( ThinPlateSplineTransformPerformanceTest "" "Common"
  ${TestDataDir}/parameters_TPSTransformTest.txt
  ${elastix_BINARY_DIR}/Testing )
elx_add_test( ThinPlateSplineTransformTest "" "Common"
  ${TestDataDir}/parameters_TPSTransformTest.txt )
elx_add_test( AdvanceOneStepParallellizationTest "" "Common" )
elx_add_test( AccumulateDerivativesParallellizationTest "" "Common" )
elx_add_test( BSplineTransformPointPerformanceTest "" "Common"
  ${TestDataDir}/parameters_AdvancedBSplineDeformableTransformTest.txt )
elx_add_test( BSplineJacobianGradientPerformanceTest "" "Common"
  ${TestDataDir}/parameters_AdvancedBSplineDeformableTransformTest.txt )

# Add tests that run OpenCL
if( ELASTIX_USE_OPENCL )
  # OpenCL core tests
  elx_add_opencl_test( OpenCLBufferTest "" "OpenCL core" "OpenCLBufferTest.cl" )
  elx_add_opencl_test( OpenCLContextTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLDeviceTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLEventTest "" "OpenCL core" "OpenCLEventTest.cl" )
  elx_add_opencl_test( OpenCLImageTest "" "OpenCL core" "OpenCLImageTest.cl" )
  elx_add_opencl_test( OpenCLKernelManagerTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLKernelTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLKernelToImageBridgeTest "" "OpenCL core" "OpenCLKernelToImageBridgeTest.cl" )
  elx_add_opencl_test( OpenCLPlatformTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLProfilingTimeProbeTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLSamplerTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLSimpleTest "" "OpenCL core" "OpenCLSimpleTest1.cl;OpenCLSimpleTest2.cl" )
  elx_add_opencl_test( OpenCLSizeTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLStringUtilsTest "" "OpenCL core" "" )
  elx_add_opencl_test( OpenCLVectorTest "" "OpenCL core" "OpenCLVectorTest.cl" )

  # OpenCL filters tests
  elx_add_opencl_test( GPUFactoriesTest "" "OpenCL" "" )

  elx_add_opencl_test( GPUBSplineDecompositionImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_decomposition_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_decomposition_GPU.mha )

  elx_add_opencl_test( GPUCastImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_cast_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_cast_GPU.mha )

  elx_add_opencl_test( GPUShrinkImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_shrink_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_shrink_GPU.mha )

  elx_add_opencl_test( GPURecursiveGaussianImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_recursive_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_recursive_GPU.mha )

  elx_add_opencl_test( GPUSmoothingRecursiveGaussianImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_smooth_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_smooth_GPU.mha )

  elx_add_opencl_test( GPUGenericMultiResolutionPyramidImageFilterTest "" "OpenCL" ""
    ${TestDataDir}/3DCT_lung_baseline.mha
    ${TestOutputDir}/3DCT_lung_baseline_generic_CPU.mha
    ${TestOutputDir}/3DCT_lung_baseline_generic_GPU.mha )

  # Affine transform tests
  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestAffine" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_affine_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_affine_GPU.mha
    -i   NearestNeighbor
    -t   Affine
    -rmse 2.8 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-LinearAffine" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_linear_affine_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_linear_affine_GPU.mha
    -i   Linear
    -t   Affine
    -rmse 1.0 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-BSplineAffine" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_affine_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_affine_GPU.mha
    -i   BSpline
    -t   Affine
    -rmse 1.0 )

  # Translation transform tests
  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestTranslation" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_translation_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_translation_GPU.mha
    -i   NearestNeighbor
    -t   Translation
    -rmse 0.0 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-LinearTranslation" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_linear_translation_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_linear_translation_GPU.mha
    -i   Linear
    -t   Translation
    -rmse 0.1 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-BSplineTranslation" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_translation_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_translation_GPU.mha
    -i   BSpline
    -t   Translation
    -rmse 0.11 )

  # BSpline transform tests
  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestBSpline" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_bspline_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_bspline_GPU.mha
    -i   NearestNeighbor
    -t   BSpline
    -p   ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt
    -rmse 0.5 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-LinearBSpline" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_linear_bspline_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_linear_bspline_GPU.mha
    -i   Linear
    -t   BSpline
    -p   ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt
    -rmse 0.03 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-BSplineBSpline" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_bspline_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_bspline_GPU.mha
    -i   BSpline
    -t   BSpline
    -p   ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt
    -rmse 0.04 )

  # Euler transform tests
  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestEuler" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_euler_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_euler_GPU.mha
    -i   NearestNeighbor
    -t   Euler
    -rmse 0.0 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-LinearEuler" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_linear_euler_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_linear_euler_GPU.mha
    -i   Linear
    -t   Euler
    -rmse 0.1 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-BSplineEuler" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_euler_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_euler_GPU.mha
    -i   BSpline
    -t   Euler
    -rmse 0.11 )

  # Similarity transform tests
  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestSimilarity" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_similarity_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_nearest_similarity_GPU.mha
    -i   NearestNeighbor
    -t   Similarity
    -rmse 0.0 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-LinearSimilarity" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_linear_similarity_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_linear_similarity_GPU.mha
    -i   Linear
    -t   Similarity
    -rmse 0.1 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-BSplineSimilarity" "OpenCL" ""
    -in  ${TestDataDir}/3DCT_lung_baseline.mha
    -out ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_similarity_CPU.mha
         ${TestOutputDir}/3DCT_lung_baseline_resample_bspline_similarity_GPU.mha
    -i   BSpline
    -t   Similarity
    -rmse 0.1 )

  elx_add_opencl_test( GPUResampleImageFilterTest "-NearestComboAffine" "OpenCL" ""
     -in  ${TestDataDir}/3DCT_lung_baseline.mha
     -out ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_affine_CPU.mha
          ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_affine_GPU.mha
     -i   NearestNeighbor
     -t   Affine
     -c -rmse 2.8 )

   elx_add_opencl_test( GPUResampleImageFilterTest "-NearestComboBSpline" "OpenCL" ""
     -in  ${TestDataDir}/3DCT_lung_baseline.mha
     -out ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_bspline_CPU.mha
          ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_bspline_GPU.mha
     -i   NearestNeighbor
     -t   BSpline
     -p   ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt
     -c -rmse 0.0 )

   elx_add_opencl_test( GPUResampleImageFilterTest "-NearestComboAffineBSpline" "OpenCL" ""
     -in  ${TestDataDir}/3DCT_lung_baseline.mha
     -out ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_affine_bspline_CPU.mha
          ${TestOutputDir}/3DCT_lung_baseline_resample_combo_nearest_affine_bspline_GPU.mha
     -i   NearestNeighbor
     -t   Affine BSpline
     -p   ${elastix_SOURCE_DIR}/Testing/parameters_AdvancedBSplineDeformableTransformTest.txt
     -c -rmse 0.3 )

endif()

#---------------------------------------------------------------------
# Add tests that run elastix

# Run 2D brain example test
#
# This is just a consistency test over time, since the registration result
# differs from machine to machine.

# Select machine dependent baseline image, basename.mha
# The path will be stored in variable "baselineImage_${basename}"
function( find_image_baseline basename )
  set( basevar baselineImage_${basename} )
  find_file( ${basevar} NAMES ${basename}.mha PATHS ${TestSiteBaselineDir} ${TestBaselineDir} NO_DEFAULT_PATH )
endfunction()

find_image_baseline( example )

if( knownChecksumSystem OR specializedTransformParameterSystem )
  if( ( SITE STREQUAL "LKEB-ELDB91" ) OR (SITE STREQUAL "LKEB.MacMini") OR (SITE STREQUAL "BIGR.cluster") OR (SITE STREQUAL "BIGR.PCStefan") )
    set( baselineImage_example ${TestBaselineDir}/example_${SITE}.mha )
  endif()
endif()

# Run the 2D brain example registration
elx_add_run_test( example
  "IMAGE" ${baselineImage_example}
  -f ${ELASTIX_DOX_DIR}/exampleinput/fixed.mhd
  -m ${ELASTIX_DOX_DIR}/exampleinput/moving.mhd
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_Rigid.txt
  -p ${ELASTIX_DOX_DIR}/exampleinput/parameters_BSpline.txt )

# Run 3D registration with a 'common' parameter file:
find_image_baseline( 3DCT_lung )
elx_add_run_test( 3DCT_lung.example
  "IMAGE;PARAMETERS;OVERLAP;LANDMARKS" ${baselineImage_3DCT_lung}
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001.txt )

# Test some transforms
elx_add_run_test( 3DCT_lung.NC.translation.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.translation.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.euler.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.euler.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.affine.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -p ${TestDataDir}/parameters.3D.NC.affine.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.bspline_r.ASGD.001a
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline_r.ASGD.001a.txt )

# Test some metrics
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.NMI.bspline.ASGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NMI.bspline.ASGD.001.txt )

elx_add_run_test( 3DCT_lung.Kappa.bspline.ASGD.001
  "CHECKSUM;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline_mask.mha
  -m ${TestDataDir}/3DCT_lung_followup_mask.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.Kappa.bspline.ASGD.001.txt )

# Test some combo metrics
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.002
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fp ${TestDataDir}/3DCT_lung_baseline.txt
  -mp ${TestDataDir}/3DCT_lung_followup.txt
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.003
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.003.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.004
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.004.txt )

configure_file(
  ${TestDataDir}/parameters.3D.NC.bspline.ASGD.004b.txt.in
  ${TestOutputDir}/parameters.3D.NC.bspline.ASGD.004b.txt @ONLY )
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.004b
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestOutputDir}/parameters.3D.NC.bspline.ASGD.004b.txt )

# Test some optimizers
elx_add_run_test( 3DCT_lung.NC.bspline.SGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.QN.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.QN.001.txt )

# Test some samplers
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.002
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.002.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.003
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.003.txt )

elx_add_run_test( 3DCT_lung.MI.bspline.SGD.004
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -fMask ${TestDataDir}/3DCT_lung_baseline_mask.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.004.txt )

# Test some interpolators
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.002
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.002.txt )

elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.003
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.003.txt )

# Test multi-threading effects for SSD
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001-Threads1
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt
  -threads 1 )
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001-Threads2
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt
  -threads 2 )
elx_add_run_test( 3DCT_lung.SSD.bspline.ASGD.001-Threads4
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.SSD.bspline.ASGD.001.txt
  -threads 4 )

# Test multi-threading effects for NC
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001a-Threads1
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001a.txt
  -threads 1 )
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001a-Threads2
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001a.txt
  -threads 2 )
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001a-Threads4
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001a.txt
  -threads 4 )

# Test multi-threading effects for MI
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001-Threads1
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt
  -threads 1 )
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001-Threads2
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt
  -threads 2 )
elx_add_run_test( 3DCT_lung.MI.bspline.SGD.001-Threads4
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.MI.bspline.SGD.001.txt
  -threads 4 )

# Test several ASGD options
elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001a # auto estimation and adaptive stepsize
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001a.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001b # auto estimation and no adaptive stepsize
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001b.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001c # manual estimation and adaptive stepsize
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001c.txt )

elx_add_run_test( 3DCT_lung.NC.bspline.ASGD.001d # manual estimation and no adaptive stepsize
  "CHECKSUM;PARAMETERS;OVERLAP;LANDMARKS"
  -f ${TestDataDir}/3DCT_lung_baseline.mha
  -m ${TestDataDir}/3DCT_lung_followup.mha
  -t0 ${TestDataDir}/transformparameters.3DCT_lung.affine.txt
  -p ${TestDataDir}/parameters.3D.NC.bspline.ASGD.001d.txt )

# Test transformix to check memory problem
trx_add_test( TransformixMemoryTest
  -in ${TestDataDir}/3DCT_lung_baseline_small.mha
  -tp ${TestDataDir}/transformparameters.3DCT_lung.affine.txt )

