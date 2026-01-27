set(ELX_COMMON_GTEST_SOURCES
  ../../Core/Main/GTesting/elxCoreMainGTestUtilities.h
  ../../Core/Main/GTesting/elxCoreMainGTestUtilities.cxx
  elxConversionGTest.cxx
  elxDefaultConstructGTest.cxx
  elxElastixMainGTest.cxx
  elxGTestUtilities.h
  elxResampleInterpolatorGTest.cxx
  elxResamplerGTest.cxx
  elxTransformIOGTest.cxx
  itkAdvancedImageToImageMetricGTest.cxx
  itkAdvancedMeanSquaresImageToImageMetricGTest.cxx
  itkComputeImageExtremaFilterGTest.cxx
  itkCorrespondingPointsEuclideanDistancePointMetricGTest.cxx
  itkGridScheduleComputerGTest.cxx
  itkImageFileCastWriterGTest.cxx
  itkImageFullSamplerGTest.cxx
  itkImageGridSamplerGTest.cxx
  itkImageRandomCoordinateSamplerGTest.cxx
  itkImageRandomSamplerGTest.cxx
  itkImageRandomSamplerSparseMaskGTest.cxx
  itkImageSamplerGTest.cxx
  itkParameterMapInterfaceTest.cxx
)

if(USE_ImpactMetric)
  list(APPEND ELX_COMMON_GTEST_SOURCES elxImpactMetricGTest.cxx)
endif()

add_executable(CommonGTest ${ELX_COMMON_GTEST_SOURCES})

target_compile_definitions(CommonGTest PRIVATE
  _USE_MATH_DEFINES # For M_PI.
  ELX_CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  ELX_CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

if(USE_ImpactMetric)
  find_package(Torch REQUIRED)
  target_link_libraries(CommonGTest
   GTest::GTest GTest::Main
    ${ITK_LIBRARIES}
    elastix_lib
    "${TORCH_LIBRARIES}")
else()
  target_link_libraries(CommonGTest
    GTest::GTest GTest::Main
    ${ITK_LIBRARIES}
    elastix_lib
    )
endif()

add_test(NAME CommonGTest_test COMMAND CommonGTest)
