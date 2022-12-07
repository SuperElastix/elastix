project(ANN)

# Compile time option to select wheter or not to accept self matches
# during nearest neighbour search. The default is false for backward
# compatibility.
mark_as_advanced(ELASTIX_KNN_ALLOW_SELF_MATCH)
set(ELASTIX_KNN_ALLOW_SELF_MATCH false
  CACHE BOOL "Allow self matches during kNN search")
set(ANN_ALLOW_SELF_MATCH "ANNfalse")
if(${ELASTIX_KNN_ALLOW_SELF_MATCH})
  set(ANN_ALLOW_SELF_MATCH "ANNtrue")
endif()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ANN/ANN.h.in
  ${elastix_BINARY_DIR}/ANN/ANN.h
  @ONLY)

# Include these directories to look for header files.
include_directories(
  ${ANN_SOURCE_DIR}/include
  ${ANN_SOURCE_DIR}/src
)

# Sources of non-templated classes.
set(ANN_SRCS
  src/ANN.cpp
  src/bd_fix_rad_search.cpp
  src/bd_pr_search.cpp
  src/bd_search.cpp
  src/bd_tree.cpp
  src/brute.cpp
  src/kd_dump.cpp
  src/kd_fix_rad_search.cpp
  src/kd_pr_search.cpp
  src/kd_search.cpp
  src/kd_split.cpp
  src/kd_tree.cpp
  src/kd_util.cpp
  src/perf.cpp
)

# Create the ANNlib library
add_library(ANNlib SHARED ${ANN_SRCS})

include(GenerateExportHeader)
generate_export_header(ANNlib
  EXPORT_FILE_NAME ${elastix_BINARY_DIR}/ANN/ANNExport.h)

# Group in IDE's like Visual Studio
set_property(TARGET ANNlib PROPERTY FOLDER "libraries")
