# Example project for using elastix code from external projects.
project(elxExternalProject)

# Minimum CMake version. Intended to correspond with the `cmake_minimum_required` version of elastix.
cmake_minimum_required(VERSION 3.16.3)

find_package(Elastix REQUIRED)

# Use the version of ITK from Elastix.
if ( DEFINED ELASTIX_ITK_DIR)
  set(ITK_DIR "${ELASTIX_ITK_DIR}" CACHE PATH "ITK_DIR from Elastix" FORCE)
endif()
find_package(ITK REQUIRED)

# Use the version of Torch from Elastix.
if ( DEFINED ELASTIX_Torch_DIR)
  set(Torch_DIR "${ELASTIX_Torch_DIR}" CACHE PATH "Torch_DIR from Elastix" FORCE)
endif()

if ( DEFINED Torch_DIR)
  find_package(Torch REQUIRED)
endif()


include(${ELASTIX_CONFIG_TARGETS_FILE})

# Build a small example executable.
add_executable(elastix_translation_example ElastixTranslationExample.cxx)

set_property(TARGET elastix_translation_example PROPERTY CXX_STANDARD 17)

target_include_directories(elastix_translation_example
  PRIVATE ${ELASTIX_INCLUDE_DIRS} ${ITK_INCLUDE_DIRS})

target_link_libraries(elastix_translation_example
  PRIVATE ${ITK_LIBRARIES} elastix_lib)

# Build the example executable using Impact metric
add_executable(elastix_impact_metric_example ElastixImpactMetricExample.cxx)

set_property(TARGET elastix_impact_metric_example PROPERTY CXX_STANDARD 17)

target_include_directories(elastix_impact_metric_example
  PRIVATE ${ELASTIX_INCLUDE_DIRS} ${ITK_INCLUDE_DIRS})

target_link_libraries(elastix_impact_metric_example
  PRIVATE ${ITK_LIBRARIES} elastix_lib)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Identity_2D.pt
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
