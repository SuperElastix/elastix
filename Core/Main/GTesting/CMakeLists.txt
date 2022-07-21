add_executable(ElastixLibGTest
  elxCoreMainGTestUtilities.h
  elxCoreMainGTestUtilities.cxx
  ElastixLibGTest.cxx
  itkElastixRegistrationMethodGTest.cxx
  itkTransformixFilterGTest.cxx
)

target_link_libraries(ElastixLibGTest
  GTest::GTest
  GTest::Main
  elastix_lib
  transformix_lib
  ${ITK_LIBRARIES}
)

target_compile_definitions(ElastixLibGTest PRIVATE ELX_CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" ELX_CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")

if(ELASTIX_USE_OPENCL)
  target_link_libraries(ElastixLibGTest elxOpenCL)
endif()

add_test(NAME ElastixLibGTest_test COMMAND ElastixLibGTest)
