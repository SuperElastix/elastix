
CUDA_ADD_LIBRARY( elastix_cuda cudaResampleImageFilter.cu cudaResampleImageFilter.cuh )
CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/C/common/inc )

# Add some additional options to the nvcc compiler
SET( CUDA_NVCC_FLAGS "--ptxas-options=-v;-use_fast_math;-gencode=arch=compute_11,code=\\\"sm_11,compute_11\\\";-gencode=arch=compute_13,code=\\\"sm_13,compute_13\\\";-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"" CACHE STRING "Over original CUDA_NVCC_FLAGS" FORCE )

# Group in IDE's like Visual Studio
SET_PROPERTY( TARGET elastix_cuda PROPERTY FOLDER "libraries" )

#---------------------------------------------------------------------
#
# Try to install the CUDA runtime library
#
#
# A CUDA installation always has a 32 bit and a 64 bit dll.
# The current build may be 32 or 64 bit.
# For linux the library name is stored in CUDA_CUDART_LIBRARY.
# For Windows this variable contains the lib, while we need the dll.

IF( WIN32 )
  # STEP 1: Try to determine if this is a 32 or 64 bit build
  #
  # Depend on CUDA_64_BIT_DEVICE_CODE, which checks
  #   IF( CMAKE_SIZEOF_VOID_P EQUAL 8 ) -> 64 bit, otherwise 32
  SET( systembuild64 ${CUDA_64_BIT_DEVICE_CODE} )


  # STEP 2: Get the full names of the cudart libraries
  #
  # The cudart library can be found under the CUDA_TOOLKIT_ROOT_DIR
  # On my windows machine:
  #   C:/NVIDIA/CUDA/bin/cudart32_30_14.dll
  #   C:/NVIDIA/CUDA/bin64/cudart64_30_14.dll
  # On my Ubuntu linux virtual machine:
  #   /usr/local/cuda/lib/libcudart.so
  #   /usr/local/cuda/lib64/libcudart.so
  # I don't know what the path is on other architectures.

  # Find all files with cudart and the shared library
  # prefix&suffix in the name
  FILE( GLOB_RECURSE cudartlibnames
    ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudart*${CMAKE_SHARED_LIBRARY_SUFFIX} )
  # On Windows also cudartemu files exist
  FILE( GLOB_RECURSE cudartemulibnames
    ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudartemu*${CMAKE_SHARED_LIBRARY_SUFFIX} )
  # Remove the cudartemu files from the list
  LIST( LENGTH cudartemulibnames noemus )
  IF( NOT ${noemus} EQUAL 0 )
    LIST( REMOVE_ITEM cudartlibnames ${cudartemulibnames} )
  ENDIF()


  # STEP 3: Determine which lib we need to install
  #
  # We decide based on the name of the relative path to the library.
  # On my windows machine:
  #   bin/cudart32_30_14.dll
  #   bin64/cudart64_30_14.dll
  # On my Ubuntu linux virtual machine:
  #   lib/libcudart.so
  #   lib64/libcudart.so
  # So use the rule that if we need 64 bit you have three characters
  # followed by 64, and if we need 32 bit we have three characters
  # followed by /.
  foreach( cudartlib ${cudartlibnames} )
    # Get path relative to CUDA_TOOLKIT_ROOT_DIR
    FILE( RELATIVE_PATH rel_path ${CUDA_TOOLKIT_ROOT_DIR} ${cudartlib} )

    # Get the fourth character
    string( SUBSTRING ${rel_path} 3 1 fourthcharacter )

    # For 64-bit builds assume:
    #   A) three characters followed by "64"
    #   B) ${systembuild64} is TRUE
    IF( ${rel_path} MATCHES "...64" AND ${systembuild64} )
      SET( cudartlibtoinstall ${cudartlib} )
    # For 32-bit builds assume:
    #   A) three characters followed by "/"
    #   B) ${systembuild64} is FALSE
    ELSEIF( ${fourthcharacter} STREQUAL "/" AND NOT ${systembuild64} )
      SET( cudartlibtoinstall ${cudartlib} )
    ENDIF()
  endforeach()

ELSE( WIN32 )
  # If not Windows, assume linux
  # Linux, however, uses symbolic links:
  #   libcudart.so -> libcudart.so.3 -> libcudart.so.3.1.9
  # The last one is the real library that is needed.
  # the linker, however, seems to link against the middle name

  # Get libcudart.so
  SET( cudartlib ${CUDA_CUDART_LIBRARY} )

  # Get libcudart.so.3.1.9 by resolving all symbolic links
  GET_FILENAME_COMPONENT( cudartlibtoinstall ${cudartlib} REALPATH )

  # Get libcudart.so.3, since that is the linked name appearantly
  FILE( GLOB cudartlibrenamed ${cudartlib}.? )
  GET_FILENAME_COMPONENT( cudartlibrenamed ${cudartlibrenamed} NAME )

ENDIF( WIN32 )


# STEP 4: Install the CUDA runtime library
#
IF( WIN32 )
  INSTALL( FILES ${cudartlibtoinstall}
    DESTINATION . )
    #COMPONENT libraries )
ELSE()
  INSTALL( FILES ${cudartlibtoinstall}
    DESTINATION lib
    #COMPONENT libraries
    #NAMELINK_ONLY )
    RENAME ${cudartlibrenamed} )
ENDIF()

