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
  # If the user set USE_CUDAResampler ON, but ELASTIX_USE_CUDA was OFF,
  # then issue a warning.
  IF( USE_CUDAResampler )
    MESSAGE( WARNING "You selected to compile CUDAResampler, "
      "but ELASTIX_USE_CUDA is OFF.\n"
      "Set both options to ON to be able to build this component." )
  ENDIF()

  # If ELASTIX_USE_CUDA is not selected, then the elastix_cuda
  # library is not created, and we cannot compile this component.
  SET( USE_CUDAResampler OFF CACHE BOOL "Compile this component" FORCE )

  # This is required to get the CUDAResampler out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( CUDAResampler )
ENDIF()
