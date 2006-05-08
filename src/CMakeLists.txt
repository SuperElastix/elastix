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
${elastix_SOURCE_DIR}/Common/xout
${elastix_SOURCE_DIR}/Common/LineSearchOptimizers)

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
# Set correct Linker options for GCC. Only tested for Linux.

IF(CMAKE_COMPILER_IS_GNUCXX)
  # -W1,-E means that the dynamic symbol table is exported. The component can now
	# access the symbols of the executable.
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}	-W1,-E")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

#--------------------------------------------------------------------
# Make sure the GDCM lib is not used. It causes segmentation faults.

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(ELX_GDCMHACK ${elastix_SOURCE_DIR}/Common/itkGDCMImageIOFactory.cxx   CACHE INTERNAL "Hack to avoid the GDCMlib")
ELSE(CMAKE_COMPILER_IS_GNUCXX)
  SET(ELX_GDCMHACK CACHE INTERNAL "Hack to avoid the GDCMlib")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

#---------------------------------------------------------------------
# Make sure the MS VC compiler has enough memory 
# stil necessary?? No, it doesn't seem so.

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

#--------------------------------------------------------------------
# Configure the examples

SET(ELX_DOX_DIR ${elastix_SOURCE_DIR}/../dox )
SET(ELX_HELP_DIR ${elastix_SOURCE_DIR}/../help CACHE PATH "path to the doxygen generated help files and the examples")

# Copy the examples to the help directory

IF(WIN32)
  CONFIGURE_FILE(
	  ${ELX_DOX_DIR}/example_windows.bat
		${ELX_HELP_DIR}/example_windows.bat
		COPYONLY )
ELSE(WIN32)
  CONFIGURE_FILE(
	  ${ELX_DOX_DIR}/example
		${ELX_HELP_DIR}/example
		COPYONLY )
ENDIF(WIN32)

MAKE_DIRECTORY( ${ELX_HELP_DIR}/exampleinput )
MAKE_DIRECTORY( ${ELX_HELP_DIR}/exampleoutput )

SET(ELX_EXAMPLEINPUTFILES
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
solution_deformedmovingimage.raw )

FOREACH( ELX_EXAMPLEINPUTFILE ${ELX_EXAMPLEINPUTFILES} )
  CONFIGURE_FILE(
	  ${ELX_DOX_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
		${ELX_HELP_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
		COPYONLY )
ENDFOREACH( ELX_EXAMPLEINPUTFILE )


#---------------------------------------------------------------------
# Configure the doxygen-configuration

FIND_PACKAGE(Doxygen QUIET)
STRING(COMPARE NOTEQUAL ${DOXYGEN} "DOXYGEN-NOTFOUND" Doxygen_FOUND)

IF(Doxygen_FOUND)

  # Set the path to the doxygen configuration source
  SET(ELX_DOXYGEN_DIR ${ELX_DOX_DIR}/doxygen)

  # Get the version number of doxygen
	EXEC_PROGRAM(${DOXYGEN} ARGS "--version" OUTPUT_VARIABLE ELX_DOXYGEN_VERSION)
	
	# Get date and time
	IF(UNIX)
		EXEC_PROGRAM(date OUTPUT_VARIABLE ELX_DOXYGEN_DATE)
	ENDIF(UNIX)
	IF(WIN32)
		SET(ELX_DOXYGEN_DATE "Sun Jan 1 00:00:00 2006")
	ENDIF(WIN32)

	# Get the version number of elastix
	IF(UNIX)
		SET(ELX_BASECOMPONENT_FILE "${elastix_SOURCE_DIR}/Core/Install/elxBaseComponent.h")
		EXEC_PROGRAM(grep ARGS "\"#define __ELASTIX_VERSION\"" "${ELX_BASECOMPONENT_FILE}" OUTPUT_VARIABLE ELX_VERSION)
		STRING(SUBSTRING ${ELX_VERSION} 26 3 ELX_VERSION)
	ENDIF(UNIX)
	
	# Configure the doxygen configuration
	CONFIGURE_FILE(
	  ${ELX_DOXYGEN_DIR}/doxyfile.in
		${ELX_HELP_DIR}/doxyfile.out @ONLY )
	# Configure the footer of the help website.
	CONFIGURE_FILE(${elastix_SOURCE_DIR}/../dox/doxygen/DoxygenFooter.html.in ${ELX_HELP_DIR}/DoxygenFooter.html @ONLY)
	
ENDIF(Doxygen_FOUND)


