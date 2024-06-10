# Example project for using elastix code from external projects.
project(elxExternalProject)

# Minimum CMake version. Intended to correspond with the `cmake_minimum_required` version of elastix.
cmake_minimum_required(VERSION 3.16.3)

if (MSVC)
  # See "`MSVC_VERSION` updated to 1940 for vs2022 17.10", https://gitlab.kitware.com/cmake/cmake/-/issues/26015
  if(MSVC_VERSION GREATER_EQUAL 1940)
    # Avoid segfaults triggered by the ABI change of having a constexpr mutex constructor.
    # See https://github.com/microsoft/STL/wiki/Changelog#vs-2022-1710
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
  endif()
endif()

find_package(Elastix REQUIRED)

# Elastix might use OpenMP.
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Use the version of ITK from Elastix.
if ( DEFINED ELASTIX_ITK_DIR)
  set(ITK_DIR "${ELASTIX_ITK_DIR}" CACHE PATH "ITK_DIR from Elastix" FORCE)
endif()
find_package(ITK REQUIRED)

include(${ELASTIX_CONFIG_TARGETS_FILE})

# Build a small example executable.
add_executable(elastix_translation_example ElastixTranslationExample.cxx)

set_property(TARGET elastix_translation_example PROPERTY CXX_STANDARD 17)

target_include_directories(elastix_translation_example
  PRIVATE ${ELASTIX_INCLUDE_DIRS} ${ITK_INCLUDE_DIRS})

target_link_libraries(elastix_translation_example
  PRIVATE ${ITK_LIBRARIES} elastix_lib)
