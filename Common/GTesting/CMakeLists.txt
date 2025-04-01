add_executable(CommonGTest
  ../../Core/Main/GTesting/elxCoreMainGTestUtilities.h
  ../../Core/Main/GTesting/elxCoreMainGTestUtilities.cxx
  elxConversionGTest.cxx
  elxDefaultConstructGTest.cxx
  elxElastixMainGTest.cxx
  elxGTestUtilities.h
  elxImpactMetricGTest.cxx
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

target_compile_definitions(CommonGTest PRIVATE
  _USE_MATH_DEFINES # For M_PI.
  ELX_CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  ELX_CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  elastix_lib
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
