add_executable(CommonGTest
  itkImageMaskSpatialObject2GTest.cxx
  )
target_link_libraries(CommonGTest
  GTest::GTest GTest::Main
  ${ITK_LIBRARIES}
  )
add_test(NAME CommonGTest_test COMMAND CommonGTest)
