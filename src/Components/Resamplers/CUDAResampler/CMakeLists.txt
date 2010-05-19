IF(ELASTIX_USE_CUDA)
  ADD_ELXCOMPONENT( CUDAResampler
   elxCUDAResampler.h
   elxCUDAResampler.hxx
   elxCUDAResampler.cxx
   itkCUDAResampleImageFilter.h
   itkCUDAResampleImageFilter.hxx )

  TARGET_LINK_LIBRARIES( CUDAResampler elastix_cuda )

  INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )
ENDIF(ELASTIX_USE_CUDA)
