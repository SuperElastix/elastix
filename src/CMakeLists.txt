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
# Include directories

SET( elxCommon_INCLUDE_DIRECTORIES
 ${elastix_SOURCE_DIR}/Common
 ${elastix_SOURCE_DIR}/Common/CostFunctions
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

SET( ELX_DOX_DIR  ${elastix_SOURCE_DIR}/../dox )
SET( ELX_HELP_DIR ${elastix_SOURCE_DIR}/../help CACHE PATH
  "path to the doxygen generated help files and the examples" )

# Copy the examples to the help directory

IF( WIN32 )
  CONFIGURE_FILE(
    ${ELX_DOX_DIR}/example.bat
    ${ELX_HELP_DIR}/example.bat
    COPYONLY )
ELSE( WIN32 )
  CONFIGURE_FILE(
    ${ELX_DOX_DIR}/example
    ${ELX_HELP_DIR}/example
    COPYONLY )
ENDIF( WIN32 )

MAKE_DIRECTORY( ${ELX_HELP_DIR}/exampleinput )
MAKE_DIRECTORY( ${ELX_HELP_DIR}/exampleoutput )

SET( ELX_EXAMPLEINPUTFILES
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
    ${ELX_DOX_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    ${ELX_HELP_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    COPYONLY )
ENDFOREACH( ELX_EXAMPLEINPUTFILE )

#---------------------------------------------------------------------
# Configure the doxygen-configuration

FIND_PACKAGE( Doxygen QUIET )
STRING( COMPARE NOTEQUAL ${DOXYGEN} "DOXYGEN-NOTFOUND" Doxygen_FOUND )

IF( Doxygen_FOUND )

  # Set the path to the doxygen configuration source
  SET( ELX_DOXYGEN_DIR ${ELX_DOX_DIR}/doxygen )

  # Get the version number of doxygen
  EXEC_PROGRAM( ${DOXYGEN} ARGS "--version" OUTPUT_VARIABLE ELX_DOXYGEN_VERSION )
  
  # Get date
  IF( UNIX OR CYGWIN )
    EXEC_PROGRAM( "date '+%d-%m-%Y'" OUTPUT_VARIABLE ELX_DOXYGEN_DATE )
  ENDIF( UNIX OR CYGWIN )
  IF( WIN32 )
    IF( NOT CYGWIN )
      EXECUTE_PROCESS( COMMAND "${ELX_DOXYGEN_DIR}/doxdate.bat"
        OUTPUT_VARIABLE ELX_DOXYGEN_DATETEMP )
      STRING( REPLACE "/" "-" ELX_DOXYGEN_DATE ${ELX_DOXYGEN_DATETEMP} )
    ENDIF( NOT CYGWIN )
  ENDIF( WIN32 )

  # Get the version number of elastix
  FILE( STRINGS ${elastix_SOURCE_DIR}/Core/Install/elxBaseComponent.h
    ELX_VERSION REGEX "\(#define\ __ELASTIX_VERSION\)" )
  STRING( SUBSTRING ${ELX_VERSION} 26 3 ELX_VERSION )

  # Configure the doxygen configuration
  CONFIGURE_FILE(
    ${ELX_DOXYGEN_DIR}/doxyfile.in
    ${ELX_HELP_DIR}/doxyfile.out @ONLY )
  # Configure the footer of the help website.
  CONFIGURE_FILE(
    ${ELX_DOXYGEN_DIR}/DoxygenFooter.html.in
    ${ELX_HELP_DIR}/DoxygenFooter.html @ONLY)
  # Configure the MainPage.dox
  CONFIGURE_FILE(
    ${ELX_DOXYGEN_DIR}/MainPage.dox.in
    ${ELX_HELP_DIR}/MainPage.dox @ONLY )

ENDIF( Doxygen_FOUND )

#---------------------------------------------------------------------
# Testing

SET( BUILD_ELX_TESTING OFF CACHE BOOL
  "Perform some tests on basic functionality of elastix." )
IF( BUILD_ELX_TESTING )
  ENABLE_TESTING()
  SUBDIRS( Testing )
ENDIF( BUILD_ELX_TESTING )

