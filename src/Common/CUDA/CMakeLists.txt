
CUDA_ADD_LIBRARY( elastix_cuda cudaResampleImageFilter.cu cudaResampleImageFilter.cuh )
CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/C/common/inc )

#add some additional options to the nvcc compiler
SET( CUDA_NVCC_FLAGS "-use_fast_math;-gencode=arch=compute_11,code=\\\"sm_11,compute_11\\\";-gencode=arch=compute_20,code=\\\"sm_20,compute_20\\\"" CACHE STRING "Over original CUDA_NVCC_FLAGS" FORCE )

#ADD_LIBRARY( elastix_cuda cudaDeformationsKernel.cu )
#SET_SOURCE_FILES_PROPERTIES( cudaDeformationsKernel.cu HEADER_FILE_ONLY )

#INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )
#INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/C/common/inc )

#The DLL should be placed in the same directory as the elastix binaries
#INSTALL(TARGETS cudart32_30_14.dll)

