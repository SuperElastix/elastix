#---------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

#---------------------------------------------------------------------
PROJECT( elastix )

#---------------------------------------------------------------------
# Find ITK.
FIND_PACKAGE( ITK )
IF( ITK_FOUND )
  INCLUDE( ${ITK_USE_FILE} )
ELSE( ITK_FOUND )
  MESSAGE( FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR." )
ENDIF( ITK_FOUND )

#---------------------------------------------------------------------
# Find CUDA.
MARK_AS_ADVANCED(ELASTIX_USE_CUDA)
OPTION(ELASTIX_USE_CUDA "Use CUDA enabled GPU" OFF)

IF( ELASTIX_USE_CUDA )
  FIND_PACKAGE( CUDA )
  IF( NOT CUDA_FOUND )
    MESSAGE( FATAL_ERROR "Cannot build without CUDA.  Please set CUDA_TOOLKIT_ROOT_DIR." )
  ENDIF( NOT CUDA_FOUND )
ENDIF( ELASTIX_USE_CUDA )

# elastix depends on some ITK settings

IF( NOT ITK_USE_REVIEW )
  message( SEND_ERROR "ITK_USE_REVIEW has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
ENDIF()

IF( NOT ITK_USE_REVIEW_STATISTICS )
  message( SEND_ERROR "ITK_USE_REVIEW_STATISTICS has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
ENDIF()

# This variable seems to be internal
#IF( NOT ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE )
#  message( SEND_ERROR "ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
#ENDIF()

IF( NOT ITK_USE_ORIENTED_IMAGE_DIRECTION )
  message( SEND_ERROR "ITK_USE_ORIENTED_IMAGE_DIRECTION has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
ENDIF()

# This variable seems to be internal
#IF( NOT ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY )
# message( SEND_ERROR "ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
#ENDIF()

#---------------------------------------------------------------------
# Include directories

SET( elxCommon_INCLUDE_DIRECTORIES
 ${elastix_SOURCE_DIR}/Common
 ${elastix_SOURCE_DIR}/Common/CostFunctions
 ${elastix_SOURCE_DIR}/Common/CUDA
 ${elastix_SOURCE_DIR}/Common/ImageSamplers
 ${elastix_SOURCE_DIR}/Common/KNN
 ${elastix_SOURCE_DIR}/Common/KNN/ann_1.1/include
 ${elastix_SOURCE_DIR}/Common/LineSearchOptimizers
 ${elastix_SOURCE_DIR}/Common/ParameterFileParser
 ${elastix_SOURCE_DIR}/Common/Transforms
 ${elastix_SOURCE_DIR}/Common/xout
)

SET( elxCore_INCLUDE_DIRECTORIES
 ${elastix_SOURCE_DIR}/Core
 ${elastix_SOURCE_DIR}/Core/Install
 ${elastix_SOURCE_DIR}/Core/Kernel
 ${elastix_SOURCE_DIR}/Core/ComponentBaseClasses
 ${elastix_SOURCE_DIR}/Core/Configuration 
 ${elastix_SOURCE_DIR}/Core/Main
)

SET( elxComponents_INCLUDE_DIRECTORIES
 ${elastix_SOURCE_DIR}/Components/FixedImagePyramids
 ${elastix_SOURCE_DIR}/Components/ImageSamplers
 ${elastix_SOURCE_DIR}/Components/Interpolators
 ${elastix_SOURCE_DIR}/Components/Metrics
 ${elastix_SOURCE_DIR}/Components/MovingImagePyramids
 ${elastix_SOURCE_DIR}/Components/Optimizers
 ${elastix_SOURCE_DIR}/Components/Registrations
 ${elastix_SOURCE_DIR}/Components/ResampleInterpolators
 ${elastix_SOURCE_DIR}/Components/Resamplers
 ${elastix_SOURCE_DIR}/Components/Transforms
)

SET( elxINCLUDE_DIRECTORIES 
 ${elxCommon_INCLUDE_DIRECTORIES}
 ${elxCore_INCLUDE_DIRECTORIES}
 ${elxComponents_INCLUDE_DIRECTORIES}
 ${elastix_BINARY_DIR}
)

INCLUDE_DIRECTORIES( ${elxINCLUDE_DIRECTORIES} )

#---------------------------------------------------------------------
# Kill the anoying MS VS warning about non-safe functions.
# They hide real warnings.
IF( MSVC )
  ADD_DEFINITIONS(
    /D_SCL_SECURE_NO_DEPRECATE
    /D_CRT_SECURE_NO_DEPRECATE
    /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
    )
  # Increases address capacity
  IF ( WIN32 )
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
  ENDIF ( WIN32 )
ENDIF( MSVC )

#---------------------------------------------------------------------
# Process the sub-directories

SUBDIRS( Common )
SUBDIRS( Components )
SUBDIRS( Core )

#--------------------------------------------------------------------
# Configure the examples

SET( ELASTIX_DOX_DIR  ${elastix_SOURCE_DIR}/../dox )
SET( ELASTIX_HELP_DIR ${elastix_SOURCE_DIR}/../help CACHE PATH
  "path to the doxygen generated help files and the examples" )

# Copy the examples to the help directory

IF( WIN32 )
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/example.bat
    ${ELASTIX_HELP_DIR}/example.bat
    COPYONLY )
ELSE( WIN32 )
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/example
    ${ELASTIX_HELP_DIR}/example
    COPYONLY )
ENDIF( WIN32 )

MAKE_DIRECTORY( ${ELASTIX_HELP_DIR}/exampleinput )
MAKE_DIRECTORY( ${ELASTIX_HELP_DIR}/exampleoutput )

SET( elxEXAMPLE_INPUT_FILES
 fixed.mhd
 fixed.raw
 mask_fixed.mhd
 mask_fixed.raw
 mask_moving.mhd
 mask_moving.raw
 moving.mhd
 moving.raw
 parameters_Affine.txt
 parameters_BSpline.txt
 parameters_Rigid.txt
 parameters_Translation.txt
 solution_deformedmovingimage.mhd
 solution_deformedmovingimage.raw
)

FOREACH( ELX_EXAMPLEINPUTFILE ${ELX_EXAMPLEINPUTFILES} )
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    ${ELASTIX_HELP_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    COPYONLY )
ENDFOREACH( ELX_EXAMPLEINPUTFILE )

#---------------------------------------------------------------------
# Configure the doxygen-configuration

FIND_PACKAGE( Doxygen QUIET )
STRING( COMPARE NOTEQUAL ${DOXYGEN} "DOXYGEN-NOTFOUND" Doxygen_FOUND )

IF( Doxygen_FOUND )

  # Set the path to the doxygen configuration source
  SET( ELASTIX_DOXYGEN_DIR ${ELASTIX_DOX_DIR}/doxygen )

  # Get the version number of doxygen
  EXEC_PROGRAM( ${DOXYGEN} ARGS "--version" OUTPUT_VARIABLE ELASTIX_DOXYGEN_VERSION )
  
  # Get date
  IF( UNIX OR CYGWIN )
    EXEC_PROGRAM( "date '+%d-%m-%Y'" OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATE )
  ENDIF( UNIX OR CYGWIN )
  IF( WIN32 )
    IF( NOT CYGWIN )
      EXECUTE_PROCESS( COMMAND "${ELASTIX_DOXYGEN_DIR}/doxdate.bat"
        OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATETEMP )
      STRING( REPLACE "/" "-" ELASTIX_DOXYGEN_DATE ${ELASTIX_DOXYGEN_DATETEMP} )
    ENDIF( NOT CYGWIN )
  ENDIF( WIN32 )

  # Get the version number of elastix
  FILE( STRINGS ${elastix_SOURCE_DIR}/Core/Install/elxBaseComponent.h
    ELASTIX_VERSION REGEX "\(#define\ __ELASTIX_VERSION\)" )
  STRING( SUBSTRING ${ELASTIX_VERSION} 26 3 ELASTIX_VERSION )

  # Configure the doxygen configuration
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/doxyfile.in
    ${ELASTIX_HELP_DIR}/doxyfile.out @ONLY )
  # Configure the footer of the help website.
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/DoxygenFooter.html.in
    ${ELASTIX_HELP_DIR}/DoxygenFooter.html @ONLY)
  # Configure the MainPage.dox
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/MainPage.dox.in
    ${ELASTIX_HELP_DIR}/MainPage.dox @ONLY )

ENDIF( Doxygen_FOUND )

#---------------------------------------------------------------------
# Testing

SET( ELASTIX_BUILD_TESTING OFF CACHE BOOL
  "Perform some tests on basic functionality of elastix." )
IF( ELASTIX_BUILD_TESTING )
  ENABLE_TESTING()
  SUBDIRS( Testing )
ENDIF()

