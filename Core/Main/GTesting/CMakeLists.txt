add_executable(ElastixLibGTest
  elxCoreMainGTestUtilities.h
  ElastixFilterGTest.cxx
  ElastixLibGTest.cxx
  itkElastixRegistrationMethodGTest.cxx
)

target_link_libraries( ElastixLibGTest
  GTest::GTest
  GTest::Main
  elastix_lib
  ${ITK_LIBRARIES}
)

if( ELASTIX_USE_OPENCL )
  target_link_libraries( ElastixLibGTest elxOpenCL )
endif()

add_test(NAME ElastixLibGTest_test COMMAND ElastixLibGTest)
