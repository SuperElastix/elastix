
if( ELASTIX_USE_CUDA )
  ADD_ELXCOMPONENT( CUDAResampler
   elxCUDAResampler.h
   elxCUDAResampler.hxx
   elxCUDAResampler.cxx
   itkCUDAResampleImageFilter.h
   itkCUDAResampleImageFilter.hxx )

  if( USE_CUDAResampler )
    target_link_libraries( CUDAResampler elastix_cuda )
    include_directories( ${CUDA_TOOLKIT_INCLUDE} )
  endif()
else()
  # If the user set USE_CUDAResampler ON, but ELASTIX_USE_CUDA was OFF,
  # then issue a warning.
  if( USE_CUDAResampler )
    message( WARNING "You selected to compile CUDAResampler, "
      "but ELASTIX_USE_CUDA is OFF.\n"
      "Set both options to ON to be able to build this component." )
  endif()

  # If ELASTIX_USE_CUDA is not selected, then the elastix_cuda
  # library is not created, and we cannot compile this component.
  set( USE_CUDAResampler OFF CACHE BOOL "Compile this component" FORCE )
  mark_as_advanced( USE_CUDAResampler )

  # This is required to get the CUDAResampler out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( CUDAResampler )
endif()

