
if( ELASTIX_USE_OPENCL )
  ADD_ELXCOMPONENT( OpenCLResampler
    elxOpenCLResampler.h
    elxOpenCLResampler.hxx
    elxOpenCLResampler.cxx )

  if( USE_OpenCLResampler )
    target_link_libraries( OpenCLResampler elxOpenCL )
  endif()
else()
  # If the user set USE_OpenCLResampler ON, but ELASTIX_USE_OPENCL was OFF,
  # then issue a warning.
  if( USE_OpenCLResampler )
    message( WARNING "You selected to compile OpenCLResampler, "
      "but ELASTIX_USE_OPENCL is OFF.\n"
      "Set both options to ON to be able to build this component." )
  endif()

  # If ELASTIX_USE_OPENCL is not selected, then the elxOpenCL
  # library is not created, and we cannot compile this component.
  set( USE_OpenCLResampler OFF CACHE BOOL "Compile this component" FORCE )
  mark_as_advanced( USE_OpenCLResampler )

  # This is required to get the OpenCLResampler out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( OpenCLResampler )
endif()

