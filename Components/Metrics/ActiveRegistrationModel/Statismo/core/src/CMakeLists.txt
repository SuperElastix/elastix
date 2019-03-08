find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(ITKInternalEigen3 REQUIRED)

add_library(statismo_core STATIC ModelInfo.cxx)

target_include_directories(statismo_core PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ITK/include)

target_link_libraries(statismo_core ${Boost_LIBRARIES} ITKInternalEigen3::Eigen)

elastix_export_target(statismo_core)
# install(TARGETS statismo_core
#   ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
#   LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
#   RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
# )

