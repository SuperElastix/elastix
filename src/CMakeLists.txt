#---------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 1.8)
# This project is intended to be built outside the Insight source tree

#---------------------------------------------------------------------
PROJECT(elastix)

#---------------------------------------------------------------------
# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
          "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

#---------------------------------------------------------------------
# Include directories

SET( elxCommon_INCLUDE_DIRECTORIES
${elastix_SOURCE_DIR}/Common
${elastix_SOURCE_DIR}/Common/TransformGrouper
${elastix_SOURCE_DIR}/Common/ParameterFileParser
${elastix_SOURCE_DIR}/Common/xout )

SET( elxCore_INCLUDE_DIRECTORIES
${elastix_SOURCE_DIR}/Core
${elastix_SOURCE_DIR}/Core/Install
${elastix_SOURCE_DIR}/Core/Kernel
${elastix_SOURCE_DIR}/Core/ComponentBaseClasses
${elastix_SOURCE_DIR}/Core/Configuration 
${elastix_SOURCE_DIR}/Core/Main )

SET( elxComponents_INCLUDE_DIRECTORIES
${elastix_SOURCE_DIR}/Components/FixedImagePyramids
${elastix_SOURCE_DIR}/Components/Interpolators
${elastix_SOURCE_DIR}/Components/Metrics
${elastix_SOURCE_DIR}/Components/MovingImagePyramids
${elastix_SOURCE_DIR}/Components/Optimizers
${elastix_SOURCE_DIR}/Components/Registrations
${elastix_SOURCE_DIR}/Components/ResampleInterpolators
${elastix_SOURCE_DIR}/Components/Resamplers
${elastix_SOURCE_DIR}/Components/Transforms )

SET( elxINCLUDE_DIRECTORIES 
${elxCommon_INCLUDE_DIRECTORIES}
${elxCore_INCLUDE_DIRECTORIES}
${elxComponents_INCLUDE_DIRECTORIES} )

INCLUDE_DIRECTORIES( ${elxINCLUDE_DIRECTORIES} )

#---------------------------------------------------------------------
# Output directories.

SET (LIBRARY_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
SET (EXECUTABLE_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
SET(elastix_LIBRARY_PATH "${LIBRARY_OUTPUT_PATH}")
SET(elastix_EXECUTABLE_PATH "${EXECUTABLE_OUTPUT_PATH}")

#---------------------------------------------------------------------
# Make sure the MS VC compiler has enough memory 
# stil necessary??

#IF(CMAKE_BUILD_TOOL MATCHES "(msdev|devenv|nmake)")
# IF(CMAKE_CXX_FLAGS MATCHES "/Zm(0|1|2|3|4|5|6|7|8|9)*[^ ]")
#   STRING(REGEX REPLACE "/Zm(0|1|2|3|4|5|6|7|8|9)*[^ ]" "/Zm1600 " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# ELSE(CMAKE_CXX_FLAGS MATCHES "/Zm(0|1|2|3|4|5|6|7|8|9)*[^ ]")
#   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zm1600 ")
# ENDIF(CMAKE_CXX_FLAGS MATCHES "/Zm(0|1|2|3|4|5|6|7|8|9)*[^ ]")
#ENDIF(CMAKE_BUILD_TOOL MATCHES "(msdev|devenv|nmake)")

#---------------------------------------------------------------------
# process the sub-directories


SUBDIRS(Common)
SUBDIRS(Core)
SUBDIRS(Components)

