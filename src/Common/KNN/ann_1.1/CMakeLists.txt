PROJECT( ANN )

# Compile time option to select wheter or not to accept self matches
# during nearest neighbour search. The default is false for backward
# compatibility.
mark_as_advanced( ELASTIX_KNN_ALLOW_SELF_MATCH )
set( ELASTIX_KNN_ALLOW_SELF_MATCH false
  CACHE BOOL "Allow self matches during kNN search" )
set( ANN_ALLOW_SELF_MATCH "ANNfalse" )
if( ${ELASTIX_KNN_ALLOW_SELF_MATCH} )
  set( ANN_ALLOW_SELF_MATCH "ANNtrue" )
endif()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ANN/ANN.h.in
  ${elastix_BINARY_DIR}/ANN/ANN.h
  @ONLY )

# Include these directories to look for header files.
INCLUDE_DIRECTORIES(
  ${ANN_SOURCE_DIR}/include
  ${ANN_SOURCE_DIR}/src
)

# Sources of non-templated classes.
SET( ANN_SRCS
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
ADD_LIBRARY( ANNlib SHARED ${ANN_SRCS} )

# Necessary, because in windows dll_export is needed for creating the
# dll, but dll_import for using it in a program.
SET_TARGET_PROPERTIES( ANNlib PROPERTIES DEFINE_SYMBOL DLL_EXPORTS )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET ANNlib PROPERTY FOLDER "libraries" )

# Install
IF( WIN32 )
  INSTALL( TARGETS ANNlib
    RUNTIME DESTINATION .
    LIBRARY DESTINATION . )
    #COMPONENT libraries )
ELSE()
  INSTALL( TARGETS ANNlib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib )
    #COMPONENT libraries )
ENDIF()

