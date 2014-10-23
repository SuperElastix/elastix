# In order to build elastix with CUDA support we have to enable the following line.
#
# It is a workaround around a problem with the nvcc compiler complaining that the
# command line is to long. By default all ITK include path are also propagated to
# the nvcc compiler. These are really a lot of paths, especially with the modular
# setup in ITK4. In addition, these include paths are not needed, since the CUDA
# kernels do not use anything from the ITK.
# Therefore, we reset the include directories to nothing for this directory:
set_directory_properties( PROPERTIES INCLUDE_DIRECTORIES "" )

# Create a CUDA library containing the CUDA resampler.
#
# The compiler flag -ftemplate-depth-50 is by default propagated to the CUDA compiler.
# This, however, generates a compiler warning. We temporarily remove the flag, and add
# it again after CUDA_ADD_LIBRARY.
set( remember_flags ${CMAKE_CXX_FLAGS} )
string( REGEX REPLACE "-ftemplate-depth-50" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
CUDA_ADD_LIBRARY( elastix_cuda cudaResampleImageFilter.cu cudaResampleImageFilter.cuh )
set( CMAKE_CXX_FLAGS ${remember_flags} )

# CUDA include directories are needed
CUDA_INCLUDE_DIRECTORIES( "${CUDA_SDK_ROOT_DIR}/C/common/inc" )

# Add some additional options to the nvcc compiler
set( CUDA_NVCC_FLAGS "--ptxas-options=-v;-use_fast_math;-gencode=arch=compute_11,code=\\\"sm_11,compute_11\\\";-gencode=arch=compute_13,code=\\\"sm_13,compute_13\\\";-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"" CACHE STRING "Over original CUDA_NVCC_FLAGS" FORCE )

# Group in IDE's like Visual Studio
set_property( TARGET elastix_cuda PROPERTY FOLDER "libraries" )

#---------------------------------------------------------------------
#
# Try to install the CUDA runtime library
#
#
# A CUDA installation always has a 32 bit and a 64 bit dll.
# The current build may be 32 or 64 bit.
# For linux the library name is stored in CUDA_CUDART_LIBRARY.
# For Windows this variable contains the lib, while we need the dll.

if( WIN32 )
  # STEP 1: Try to determine if this is a 32 or 64 bit build
  #
  # Depend on CUDA_64_BIT_DEVICE_CODE, which checks
  #   if( CMAKE_SIZEOF_VOID_P EQUAL 8 ) -> 64 bit, otherwise 32
  set( systembuild64 ${CUDA_64_BIT_DEVICE_CODE} )


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
  file( GLOB_RECURSE cudartlibnames
    ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudart*${CMAKE_SHARED_LIBRARY_SUFFIX} )
  # On Windows also cudartemu files exist
  file( GLOB_RECURSE cudartemulibnames
    ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudartemu*${CMAKE_SHARED_LIBRARY_SUFFIX} )
  # Remove the cudartemu files from the list
  list( LENGTH cudartemulibnames noemus )
  if( NOT ${noemus} EQUAL 0 )
    list( REMOVE_ITEM cudartlibnames ${cudartemulibnames} )
  endif()


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
    file( RELATIVE_PATH rel_path ${CUDA_TOOLKIT_ROOT_DIR} ${cudartlib} )

    # Get the fourth character
    string( SUBSTRING ${rel_path} 3 1 fourthcharacter )

    # For 64-bit builds assume:
    #   A) three characters followed by "64"
    #   B) ${systembuild64} is TRUE
    if( ${rel_path} MATCHES "...64" AND ${systembuild64} )
      set( cudartlibtoinstall ${cudartlib} )
    # For 32-bit builds assume:
    #   A) three characters followed by "/"
    #   B) ${systembuild64} is FALSE
    elseif( ${fourthcharacter} STREQUAL "/" AND NOT ${systembuild64} )
      set( cudartlibtoinstall ${cudartlib} )
    endif()
  endforeach()

else( WIN32 )
  # If not Windows, assume linux
  # Linux, however, uses symbolic links:
  #   libcudart.so -> libcudart.so.3 -> libcudart.so.3.1.9
  # The last one is the real library that is needed.
  # the linker, however, seems to link against the middle name

  # Get libcudart.so
  set( cudartlib ${CUDA_CUDART_LIBRARY} )

  # Get libcudart.so.3.1.9 by resolving all symbolic links
  get_filename_component( cudartlibtoinstall ${cudartlib} REALPATH )

  # Get libcudart.so.3, since that is the linked name appearantly
  # Since CUDA 5 it will be libcudart.so.5.0
  file( GLOB cudartlibrenamed ${cudartlib}.? )
  list( LENGTH cudartlibrenamed cudartliblistlength )
  if( ${cudartliblistlength} EQUAL 0 )
    file( GLOB cudartlibrenamed ${cudartlib}.?.? )
  endif()
  get_filename_component( cudartlibrenamed ${cudartlibrenamed} NAME )

endif()


# STEP 4: Install the CUDA runtime library
#
if( WIN32 )
  install( FILES ${cudartlibtoinstall}
    DESTINATION . )
    #COMPONENT libraries )
else()
  install( FILES ${cudartlibtoinstall}
    DESTINATION lib
    #COMPONENT libraries
    #NAMELINK_ONLY )
    RENAME ${cudartlibrenamed} )
endif()

