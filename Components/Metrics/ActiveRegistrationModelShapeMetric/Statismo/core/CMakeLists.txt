find_package(BOOST REQUIRED)

include_directories( BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include ${Boost_INCLUDE_DIRS})

add_subdirectory(src)

if(${BUILD_TESTING})
  add_subdirectory(tests)
endif()

