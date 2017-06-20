# Example project for using elastix code from external projects.
PROJECT( elxExternalProject )

CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

# Find TIK
FIND_PACKAGE( ITK REQUIRED )
IF( ITK_FOUND )
  INCLUDE( ${ITK_USE_FILE} )
ENDIF( ITK_FOUND )

# find elastix
SET( ELASTIX_BINARY_DIR "" CACHE PATH "Path to elastix binary folder" )
SET( ELASTIX_USE_FILE ${ELASTIX_BINARY_DIR}/UseElastix.cmake )
IF( EXISTS ${ELASTIX_USE_FILE} )
 MESSAGE( STATUS "Including Elastix settings." )
 INCLUDE( ${ELASTIX_USE_FILE} )
ENDIF()

# Build a small test executable (this test is basically the same as
# the one found in the <elastix-dir>/Testing dir.
ADD_EXECUTABLE( elxtimertest itkTimerTest.cxx )

# Link to some libraries
TARGET_LINK_LIBRARIES( elxtimertest
  ITKCommon ITKBasicFilters elxCommon )



