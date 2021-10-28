add_executable(CommonGTest
  elxConversionGTest.cxx
  elxDefaultConstructibleSubclassGTest.cxx
  elxElastixMainGTest.cxx
  elxGTestUtilities.h
  elxResampleInterpolatorGTest.cxx
  elxResamplerGTest.cxx
  elxTransformIOGTest.cxx
  itkComputeImageExtremaFilterGTest.cxx
  itkParameterMapInterfaceTest.cxx
  )
target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  elastix_lib
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
