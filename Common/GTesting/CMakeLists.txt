add_executable(CommonGTest
  elxConversionGTest.cxx
  elxDefaultConstructGTest.cxx
  elxElastixMainGTest.cxx
  elxGTestUtilities.h
  elxResampleInterpolatorGTest.cxx
  elxResamplerGTest.cxx
  elxTransformIOGTest.cxx
  itkComputeImageExtremaFilterGTest.cxx
  itkImageGridSamplerGTest.cxx
  itkParameterMapInterfaceTest.cxx
  )
target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  elastix_lib
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
