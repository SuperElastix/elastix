
if(ELASTIX_USE_OPENCL)
  ADD_ELXCOMPONENT( OpenCLMovingGenericPyramid
    elxOpenCLMovingGenericPyramid.h
    elxOpenCLMovingGenericPyramid.hxx
    elxOpenCLMovingGenericPyramid.cxx)

  if(USE_OpenCLMovingGenericPyramid)
    target_include_directories(OpenCLMovingGenericPyramid PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../MovingGenericPyramid)
    target_include_directories(OpenCLMovingGenericPyramid BEFORE PRIVATE
      ${elxCommon_OpenCL_INCLUDE_DIRECTORIES}
      ${OPENCL_KERNELS_DEBUG_PATH})
    target_link_libraries(OpenCLMovingGenericPyramid elxOpenCL)
  endif()
else()
  # If the user set USE_OpenCLMovingGenericPyramid ON, but ELASTIX_USE_OPENCL was OFF,
  # then issue a warning.
  if(USE_OpenCLMovingGenericPyramid)
    message(WARNING "You selected to compile OpenCLMovingGenericPyramid, "
      "but ELASTIX_USE_OPENCL is OFF.\n"
      "Set both options to ON to be able to build this component.")
  endif()

  # If ELASTIX_USE_OPENCL is not selected, then the elxOpenCL
  # library is not created, and we cannot compile this component.
  set(USE_OpenCLMovingGenericPyramid OFF CACHE BOOL "Compile this component" FORCE)
  mark_as_advanced(USE_OpenCLMovingGenericPyramid)

  # This is required to get the OpenCLMovingGenericPyramid out of the AllComponentLibs
  # list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( OpenCLMovingGenericPyramid)
endif()
