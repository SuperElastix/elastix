
if( ELASTIX_USE_OPENCL )
  ADD_ELXCOMPONENT( OpenCLFixedGenericPyramid
    elxOpenCLFixedGenericPyramid.h
    elxOpenCLFixedGenericPyramid.hxx
    elxOpenCLFixedGenericPyramid.cxx )

  include_directories(
  ../FixedGenericPyramid )

  if( USE_OpenCLFixedGenericPyramid )
    target_link_libraries( OpenCLFixedGenericPyramid elxOpenCL )
  endif()
else()
  # If the user set USE_OpenCLFixedGenericPyramid ON, but ELASTIX_USE_OPENCL was OFF,
  # then issue a warning.
  if( USE_OpenCLFixedGenericPyramid )
    message( WARNING "You selected to compile OpenCLFixedGenericPyramid, "
      "but ELASTIX_USE_OPENCL is OFF.\n"
      "Set both options to ON to be able to build this component." )
  endif()

  # If ELASTIX_USE_OPENCL is not selected, then the elxOpenCL
  # library is not created, and we cannot compile this component.
  set( USE_OpenCLFixedGenericPyramid OFF CACHE BOOL "Compile this component" FORCE )

  # This is required to get the OpenCLFixedGenericPyramid out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( OpenCLFixedGenericPyramid )
endif()
