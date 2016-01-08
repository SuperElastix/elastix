#
# This CMakeLists file will generate the file elxSupportedImageTypes.h
# in the bin directory. This file defines the types of images for which
# elastix is compiled.
#

# User specified variables:
mark_as_advanced( ELASTIX_IMAGE_DIMENSIONS )
if( NOT ELASTIX_IMAGE_DIMENSIONS )
  set( ELASTIX_IMAGE_DIMENSIONS 2 3 4 CACHE STRING "Specify image dimensions" )
endif()

mark_as_advanced( ELASTIX_IMAGE_2D_PIXELTYPES )
if( NOT ELASTIX_IMAGE_2D_PIXELTYPES )
  set( ELASTIX_IMAGE_2D_PIXELTYPES "float" CACHE STRING "Specify 2D pixel types" )
endif()

mark_as_advanced( ELASTIX_IMAGE_3D_PIXELTYPES )
if( NOT ELASTIX_IMAGE_3D_PIXELTYPES )
  set( ELASTIX_IMAGE_3D_PIXELTYPES "short" "float" CACHE STRING "Specify 3D pixel types" )
endif()

mark_as_advanced( ELASTIX_IMAGE_4D_PIXELTYPES )
if( NOT ELASTIX_IMAGE_4D_PIXELTYPES )
  set( ELASTIX_IMAGE_4D_PIXELTYPES "short" CACHE STRING "Specify 4D pixel types" )
endif()

mark_as_advanced( USE_ALL_PIXELTYPES )
if( NOT USE_ALL_PIXELTYPES )
 option( USE_ALL_PIXELTYPES "Build all pixel types" OFF )
endif()

# Define supported dimensions and types for sanity checks.
# Gives protection against typo's.
set( supportedDimensions 2 3 4 )
set( supportedTypes "char" "unsigned char"
  "short" "unsigned short" "int" "unsigned int"
  "long" "unsigned long" "float" "double" )

if( ${USE_ALL_PIXELTYPES} )
  set( ELASTIX_IMAGE_2D_PIXELTYPES ${supportedTypes} )
  set( ELASTIX_IMAGE_3D_PIXELTYPES ${supportedTypes} )
  set( ELASTIX_IMAGE_4D_PIXELTYPES ${supportedTypes} )
endif()

# Sanity check if > 0 number of dimensions are requested
list( LENGTH ELASTIX_IMAGE_DIMENSIONS numDims )
if( ${numDims} EQUAL 0 )
  message( FATAL_ERROR "ERROR: you should set one or more values "
    " to ELASTIX_IMAGE_DIMENSIONS!\n"
    "Choose a subset of {${supportedDimensions}}." )
endif()

# Start a string containing the supported image types
# and initialize some variables.
set( supportString "" )
set( supportOpenCLImages "" )
set( index 1 )
set( numSupported 0 )

# Add supported image types to the string
foreach( dim ${ELASTIX_IMAGE_DIMENSIONS} )

  # Check dimension
  if( ${dim} EQUAL 2 )
    set( pixelTypeList ${ELASTIX_IMAGE_2D_PIXELTYPES} )
    set( whichList "ELASTIX_IMAGE_2D_PIXELTYPES" )
  elseif( ${dim} EQUAL 3 )
    set( pixelTypeList ${ELASTIX_IMAGE_3D_PIXELTYPES} )
    set( whichList "ELASTIX_IMAGE_3D_PIXELTYPES" )
  elseif( ${dim} EQUAL 4 )
    set( pixelTypeList ${ELASTIX_IMAGE_4D_PIXELTYPES} )
    set( whichList "ELASTIX_IMAGE_4D_PIXELTYPES" )
  else()
    message( FATAL_ERROR "ERROR: you selected ELASTIX_IMAGE_DIMENSIONS"
      " to include ${dim}, which is not supported!\n"
      "Choose a subset of {${supportedDimensions}}." )
  endif()

  # Sanity check if > 0 number of pixel types are requested
  list( LENGTH pixelTypeList pixelTypeListLength )
  if( ${pixelTypeListLength} EQUAL 0 )
    message( FATAL_ERROR "ERROR: you should set one or more values"
      " to ${whichList}!\n"
      "Choose a subset of {${supportedTypes}}." )
  endif()

  # Add types
  foreach( type ${pixelTypeList} )
    # Sanity check
    list( FIND supportedTypes ${type} foundIndex )
    if( ${foundIndex} EQUAL -1 )
      message( FATAL_ERROR "ERROR: you selected ${whichList}"
        " to include ${type}, which is not supported!\n"
        "Choose a subset of {${supportedTypes}}." )
    endif()

    # Add type to supportString
    set( supportString
      "${supportString}  elxSupportedImageTypeMacro( ${type}, ${dim}, ${type}, ${dim}, ${index} );\n" )

    # Increase some variables
    math( EXPR index "${index} + 1" )
    math( EXPR numSupported "${numSupported} + 1" )
  endforeach()

  # OpenCL
  if( ELASTIX_USE_OPENCL )
    # Construct supportOpenCLImages
    set( openclindex 0 )
    list( LENGTH pixelTypeList pixelTypeListLength )

    if( supportOpenCLImages STREQUAL "" )
      set( supportOpenCLImages "typedef typelist::MakeTypeList<" )
    else()
      set( supportOpenCLImages "${supportOpenCLImages}\n  typedef typelist::MakeTypeList<" )
    endif()

    foreach( type ${pixelTypeList} )
      set( supportOpenCLImages "${supportOpenCLImages}${type}" )
      math( EXPR openclindex "${openclindex} + 1" )
      if( NOT openclindex EQUAL pixelTypeListLength )
        set( supportOpenCLImages "${supportOpenCLImages}, " )
      endif()
    endforeach()
    set( supportOpenCLImages "${supportOpenCLImages}>::Type OpenCLImageTypes${dim}D;" )
  endif()

endforeach()

# Prepend the string with the number of supported images.
set( supportString
  "const unsigned int NrOfSupportedImageTypes = ${numSupported};\n\n${supportString}" )

# Put the generated string in the elxSupportedImageTypes.h
# header using configure_file.
set( UserDefinedSupportedImageTypes ${supportString} )
configure_file(
  ${elastix_SOURCE_DIR}/Core/Install/elxSupportedImageTypes.h.in
  ${elastix_BINARY_DIR}/elxSupportedImageTypes.h
  @ONLY )

# OpenCL
if( ELASTIX_USE_OPENCL )
  # Construct supportOpenCLDimensions
  set( supportOpenCLDimensions "" )

  # Check 2D
  list( FIND ELASTIX_IMAGE_DIMENSIONS 2 ELASTIX_SUPPORT_OPENCL_2D )
  if( NOT ELASTIX_SUPPORT_OPENCL_2D EQUAL -1 )
    set( tof "true" )
  else()
    set( tof "false" )
  endif()
  set( supportOpenCLDimensions "itkStaticConstMacro( Support2D, bool, ${tof} );" )

  # Check 3D
  list( FIND ELASTIX_IMAGE_DIMENSIONS 3 ELASTIX_SUPPORT_OPENCL_3D )
  if( NOT ELASTIX_SUPPORT_OPENCL_3D EQUAL -1 )
    set( tof "true" )
  else()
    set( tof "false" )
  endif()
  set( supportOpenCLDimensions
    "${supportOpenCLDimensions}\n    itkStaticConstMacro( Support3D, bool, ${tof} );" )

  set( UserDefinedSupportedDimensions "${supportOpenCLDimensions}" )
  set( UserDefinedSupportedImageTypes "${supportOpenCLImages}" )

  # Put the generated string in the elxOpenCLSupportedImageTypes.h
  # header using configure_file.
  configure_file(
    ${elastix_SOURCE_DIR}/Core/Install/elxOpenCLSupportedImageTypes.h.in
    ${elastix_BINARY_DIR}/elxOpenCLSupportedImageTypes.h
    @ONLY )
endif()

