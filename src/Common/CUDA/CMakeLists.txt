
CUDA_ADD_LIBRARY( elastix_cuda cudaResampleImageFilter.cu cudaResampleImageFilter.cuh )
CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/C/common/inc )

# Add some additional options to the nvcc compiler
SET( CUDA_NVCC_FLAGS "--ptxas-options=-v;-use_fast_math;-gencode=arch=compute_11,code=\\\"sm_11,compute_11\\\";-gencode=arch=compute_13,code=\\\"sm_13,compute_13\\\";-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"" CACHE STRING "Over original CUDA_NVCC_FLAGS" FORCE )

#ADD_LIBRARY( elastix_cuda cudaDeformationsKernel.cu )
#SET_SOURCE_FILES_PROPERTIES( cudaDeformationsKernel.cu HEADER_FILE_ONLY )

#INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )
#INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/C/common/inc )

#---------------------------------------------------------------------
# Try to install the CUDA runtime library
#
# A CUDA installation always has a 32 bit and a 64 bit dll.

message( STATUS CUDA_64_BIT_DEVICE_CODE\t${CUDA_64_BIT_DEVICE_CODE} )

# STEP 1: Try to determine if this is a 32 or 64 bit build
#
## For windows icw Visual Studio we rely on CMAKE_CL_64
## For linux we rely on the system processor name
## Otherwise, we don't know.
# Depend on CUDA_64_BIT_DEVICE_CODE, which checks
#   IF( CMAKE_SIZEOF_VOID_P EQUAL 8 ) -> 64 bit, otherwise 32
SET( systembuild64 ${CUDA_64_BIT_DEVICE_CODE} )
#SET( systembuild64 FALSE )
#IF( ${WIN32} AND ${MSVC} )
#  message( STATUS CMAKE_CL_64:\t${CMAKE_CL_64} )
# SET( systembuild64 ${CMAKE_CL_64} )
  #  IF( ${CMAKE_GENERATOR} MATCHES ".*Win64" )
  # SET( systembuild64 TRUE )
  #ENDIF()
  #ELSEIF( ${UNIX} )
  #IF ( ${CMAKE_SYSTEM_PROCESSOR} MATCHES ".*-64" )
  # SET( systembuild64 TRUE )
  #ENDIF()
  #ELSE()
  #message( WARNING "WARNING: Cannot determine if we need 32 or 64 bit library" )
  #ENDIF()

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

# Find all files with cudart and the shared library prefix&suffix in the name
FILE( GLOB_RECURSE cudartlibnames ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudart*${CMAKE_SHARED_LIBRARY_SUFFIX} )
# On Windows also cudartemu files exist
FILE( GLOB_RECURSE cudartemulibnames ${CUDA_TOOLKIT_ROOT_DIR}/*${CMAKE_SHARED_LIBRARY_PREFIX}cudartemu*${CMAKE_SHARED_LIBRARY_SUFFIX} )
# Remove the cudartemu files from the list
LIST( LENGTH cudartemulibnames noemus )
IF( NOT ${noemus} EQUAL 0 )
  LIST( REMOVE_ITEM cudartlibnames ${cudartemulibnames} )
ENDIF()

# STEP 3: Determine which lib we need to install
#
foreach( cudartlib ${cudartlibnames} )
  # Get path relative to CUDA_TOOLKIT_ROOT_DIR
  # On my windows machine:
  #   bin/cudart32_30_14.dll
  #   bin64/cudart64_30_14.dll
  # On my Ubuntu linux virtual machine:
  #   lib/libcudart.so
  #   lib64/libcudart.so
  # So use the rule that if we need 64 bit you have three characters
  # followed by 64, and if we need 32 bit we have three characters
  # followed by /.
  FILE( RELATIVE_PATH rel_path ${CUDA_TOOLKIT_ROOT_DIR} ${cudartlib} )

  # Get the fourth character
  string( SUBSTRING ${rel_path} 3 1 fourthcharacter )

  # For 64-bit builds assume:
  #   A) three characters followed by "64"
  #   B) ${systembuild64} is TRUE
  IF( ${rel_path} MATCHES "...64" AND ${systembuild64} )
    #message( STATUS Need to install\t ${cudartlib} )
    SET( cudartlibtoinstall ${cudartlib} )
  # For 32-bit builds assume:
  #   A) three characters followed by "/"
  #   B) ${systembuild64} is FALSE
  ELSEIF( ${fourthcharacter} STREQUAL "/" AND NOT ${systembuild64} )
    #message( STATUS Need to install\t ${cudartlib} )
    SET( cudartlibtoinstall ${cudartlib} )
  ENDIF()
endforeach()

# STEP 4: Install the CUDA runtime library
#
message( STATUS Need to install final\t ${cudartlibtoinstall} )
INSTALL( FILES ${cudartlibtoinstall}
  #ARCHIVE DESTINATION .
  DESTINATION .
  COMPONENT libraries )


#---------------------------------------------------------------------
# OLD STUFF:

# Find out where the CUDA dll is, by finding out about all elastix dependencies
#INCLUDE( GetPrerequisites )

#message( STATUS ${PROJECT_BINARY_DIR} )
#message( STATUS ${elastix_BINARY_DIR} )

#get_prerequisites( ${CMAKE_CURRENT_BINARY_DIR}/Release/elastix${CMAKE_EXECUTABLE_SUFFIX} prereqs 1 1 "${exepath}" "")
#FOREACH( prereq ${prereqs} )
#  IF( ${prereq} STRGREATER "cudart" )
#    gp_resolve_item( context ${prereq} exepath dirs full_lib_name )
#    INSTALL( FILES ${full_lib_name} DESTINATION . COMPONENT libraries )
#  ENDIF()
  #  INSTALL( FILES ${CUDA_TOOLKIT_ROOT_DIR}/bin64/cudart64_30_14.dll
  ##ARCHIVE DESTINATION .
  #DESTINATION .
  #COMPONENT libraries )
  #ENDFOREACH()

#gp_item_default_embedded_path( "cudart64_30_14.dll" default_embedded_path_var )
#message( STATUS ${default_embedded_path_var} )
#gp_resolve_item( context "cudart64_30_14.dll" exepath dirs resolved_item_var )
#message( STATUS ${resolved_item_var} )

