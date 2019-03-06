find_package(BOOST REQUIRED)
message(STATUS "BOOST_INCLUDE_DIRS: ${BOOST_INCLUDE_DIRS}")

add_library( statismo_core STATIC ModelInfo.cxx)
target_link_libraries(statismo_core ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(
  ${BOOST_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set_target_properties( statismo_core PROPERTIES
  DEBUG_POSTFIX "-d"
  VERSION "${statismo_LIB_VERSION}"
  SOVERSION "${statismo_LIB_SOVERSION}"
)
