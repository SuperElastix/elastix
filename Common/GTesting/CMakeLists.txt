add_executable(CommonGTest
  elxBaseComponentGTest.cxx
  elxTransformIOGTest.cxx
  itkComputeImageExtremaFilterGTest.cxx
  )
target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  elastix_lib
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
