IF( ELASTIX_USE_CUDA )
  ADD_ELXCOMPONENT( CUDAResampler
   elxCUDAResampler.h
   elxCUDAResampler.hxx
   elxCUDAResampler.cxx
   itkCUDAResampleImageFilter.h
   itkCUDAResampleImageFilter.hxx )

  IF( USE_CUDAResampler )
    TARGET_LINK_LIBRARIES( CUDAResampler elastix_cuda )
    INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_INCLUDE} )
  ENDIF()
ELSE()
  # If ELASTIX_USE_CUDA is not selected, then the elastix_cuda
  # library is not created, and we cannot compile this component.
  SET( USE_CUDAResampler OFF CACHE BOOL "Compile this component" FORCE )

  # This is required to get the CUDAResampler out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  ADD_ELXCOMPONENT( CUDAResampler elxCUDAResampler.cxx )
ENDIF()
