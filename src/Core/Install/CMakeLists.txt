#
# This CMakeLists file will generate the file elxSupportedImageTypes.h
# in the bin directory. This file defines the types of images for which
# elastix is compiled.
#

# User specified variables:
MARK_AS_ADVANCED( ELASTIX_IMAGE_DIMENSIONS )
SET( ELASTIX_IMAGE_DIMENSIONS 2 3 4
  CACHE STRING "Specify image dimensions" )

MARK_AS_ADVANCED( ELASTIX_IMAGE_2D_PIXELTYPES )
SET( ELASTIX_IMAGE_2D_PIXELTYPES "float"
  CACHE STRING "Specify 2D pixel types" )

MARK_AS_ADVANCED( ELASTIX_IMAGE_3D_PIXELTYPES )
SET( ELASTIX_IMAGE_3D_PIXELTYPES "short" "float"
  CACHE STRING "Specify 3D pixel types" )

MARK_AS_ADVANCED( ELASTIX_IMAGE_4D_PIXELTYPES )
SET( ELASTIX_IMAGE_4D_PIXELTYPES "short"
  CACHE STRING "Specify 4D pixel types" )

# Define supported dimensions and types for sanity checks.
# Gives protection against typo's.
SET( supportedDimensions 2 3 4 )
SET( supportedTypes "char" "unsigned char"
  "short" "unsigned short" "int" "unsigned int"
  "long" "unsigned long" "float" "double" )

# Start a string containing the supported image types
# and initialize some variables.
SET( supportString "" )
SET( index 1 )
SET( numSupported 0 )

# Add supported image types to the string
FOREACH( dim ${ELASTIX_IMAGE_DIMENSIONS} )

  # Check dimension
  IF( ${dim} EQUAL 2 )
    SET( pixelTypeList ${ELASTIX_IMAGE_2D_PIXELTYPES} )
    SET( whichList "ELASTIX_IMAGE_2D_PIXELTYPES" )
  ELSEIF( ${dim} EQUAL 3 )
    SET( pixelTypeList ${ELASTIX_IMAGE_3D_PIXELTYPES} )
    SET( whichList "ELASTIX_IMAGE_3D_PIXELTYPES" )
  ELSEIF( ${dim} EQUAL 4 )
    SET( pixelTypeList ${ELASTIX_IMAGE_4D_PIXELTYPES} )
    SET( whichList "ELASTIX_IMAGE_4D_PIXELTYPES" )
  ELSE()
    MESSAGE( FATAL_ERROR "WARNING: you selected ELASTIX_IMAGE_DIMENSIONS"
      " to include ${dim}, which is not supported!\n"
      "Choose one of {${supportedDimensions}}." )
  ENDIF()

  # Add types
  FOREACH( type ${pixelTypeList} )
    # Sanity check
    LIST( FIND supportedTypes ${type} foundIndex )
    IF( ${foundIndex} EQUAL -1 )
      MESSAGE( FATAL_ERROR "WARNING: you selected ${whichList}"
        " to include ${type}, which is not supported!\n"
        "Choose one of {${supportedTypes}}." )
    ENDIF()

    # Add type to supportString
    SET( supportString
      "${supportString}  elxSupportedImageTypeMacro( ${type}, ${dim}, ${type}, ${dim}, ${index} );\n" )

    # Increase some variables
    MATH( EXPR index "${index} + 1" )
    MATH( EXPR numSupported "${numSupported} + 1" )
  ENDFOREACH()
ENDFOREACH()

# Prepend the string with the number of supported images.
SET( supportString
  "const unsigned int NrOfSupportedImageTypes = ${numSupported};\n\n${supportString}" )

# Put the generated string in the elxSupportedImageTypes.h
# header using configure_file.
SET( UserDefinedSupportedImageTypes ${supportString} )
CONFIGURE_FILE(
  ${elastix_SOURCE_DIR}/Core/Install/elxSupportedImageTypes.h.in
  ${elastix_BINARY_DIR}/elxSupportedImageTypes.h
  @ONLY )

