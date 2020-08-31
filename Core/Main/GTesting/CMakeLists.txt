add_executable(ElastixLibGTest
  ElastixFilterGTest.cxx
  ElastixLibGTest.cxx
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
