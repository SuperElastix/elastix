add_executable(ElastixLibGTest
  elxCoreMainGTestUtilities.h
  elxCoreMainGTestUtilities.cxx
  ElastixLibGTest.cxx
  GetDefaultParameterMapGTest.cxx
  itkElastixRegistrationMethodGTest.cxx
  itkTransformixFilterGTest.cxx
  ParameterObjectGTest.cxx
)

target_compile_definitions(ElastixLibGTest PRIVATE
  _USE_MATH_DEFINES # For M_PI.
)

if (ITK_VERSION VERSION_LESS_EQUAL 5.4.5)
  target_link_libraries(ElastixLibGTest
    GTest::GTest
    GTest::Main
    elastix_lib
    transformix_lib
    ${ITK_LIBRARIES}
  )
else()
  target_link_libraries(ElastixLibGTest
    GTest::gtest
    GTest::gtest_main
    elastix_lib
    transformix_lib
    ${ITK_LIBRARIES}
  )
endif()


target_compile_definitions(ElastixLibGTest PRIVATE ELX_CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" ELX_CMAKE_CURRENT_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")

if(ELASTIX_USE_OPENCL)
  target_link_libraries(ElastixLibGTest elxOpenCL)
endif()

add_test(NAME ElastixLibGTest_test COMMAND ElastixLibGTest)
