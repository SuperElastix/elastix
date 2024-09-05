add_executable(CommonGTest
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
)

target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  elastix_lib
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
