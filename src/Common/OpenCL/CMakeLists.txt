
project( elastix_opencl )

# Get the full paths to the OpenCL kernels and define
# a new list of those files in the binary directory.
file( TO_NATIVE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/Kernels/
  ELASTIX_OPENCL_KERNELS_PATH )
if( WIN32 )
  # Define ITK4OPENCL_KERNELS_PATH in Windows style with back slashes
  string( REGEX REPLACE "\\\\" "\\\\\\\\" ELASTIX_OPENCL_KERNELS_PATH ${ELASTIX_OPENCL_KERNELS_PATH})
endif()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUKernels.h.in
  ${elastix_BINARY_DIR}/itkGPUKernels.h
  @ONLY )

# Get the files in the main directory
file( GLOB OpenCL_CommonFiles itk*.* )
list( APPEND OpenCL_CommonFiles ${elastix_BINARY_DIR}/itkGPUKernels.h )
source_group( "Common" FILES ${OpenCL_CommonFiles} )

# Get the files in the subdir ITKimprovements
file( GLOB OpenCL_ITKimprovementsFiles ITKimprovements/*.* )
source_group( "ITKimprovements" FILES ${OpenCL_ITKimprovementsFiles} )

# Get the files in the subdir Filters
file( GLOB OpenCL_FiltersFiles Filters/*.* )
source_group( "Filters" FILES ${OpenCL_FiltersFiles} )

# Get the files in the subdir Kernels
file( GLOB OpenCL_KernelsFiles Kernels/*.* )
source_group( "Kernels" FILES ${OpenCL_KernelsFiles} )

# Set the files, so that they are shown in the editor
SET( OpenCL_Files
  ${OpenCL_CommonFiles}
  ${OpenCL_ITKimprovementsFiles}
  ${OpenCL_FiltersFiles}
  ${OpenCL_KernelsFiles} )

# Create the openCL library
add_library( elastix_opencl ${OpenCL_Files} )

# Link it against the necessary libraries.
target_link_libraries( elastix_opencl
  ${ITK_LIBRARIES}
  ${OPENCL_LIBRARIES} )

# Group in IDE's like Visual Studio
set_property( TARGET elastix_opencl PROPERTY FOLDER "libraries" )

