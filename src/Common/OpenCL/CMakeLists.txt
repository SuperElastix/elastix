
project( elastix_opencl )

# Define directory to OpenCL Debug kernels
set( elastix_OPENCL_KERNELS_DEBUG_PATH
  ${elastix_BINARY_DIR}
  CACHE INTERNAL "Path to itkOpenCLKernels.h" )

set( elastix_OPENCL_KERNELS_DEBUG_DIR
  ${elastix_OPENCL_KERNELS_DEBUG_PATH}/elastixOpenCLDebug
  CACHE INTERNAL "Directory to OpenCL debug kernels" )

file( MAKE_DIRECTORY ${elastix_OPENCL_KERNELS_DEBUG_DIR} )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ITKimprovements/itkOpenCLKernels.h.in
  ${elastix_OPENCL_KERNELS_DEBUG_PATH}/itkOpenCLKernels.h
  @ONLY
)

# Get the files in the main directory
file( GLOB OpenCL_CommonFiles itk*.* )
list( APPEND OpenCL_CommonFiles ${elastix_OPENCL_KERNELS_DEBUG_PATH}/itkOpenCLKernels.h )

source_group( "Common" FILES ${OpenCL_CommonFiles} )

# Get the files in the subdir ITKimprovements
file( GLOB OpenCL_ITKimprovementsFiles ITKimprovements/*.* )
source_group( "ITKimprovements" FILES ${OpenCL_ITKimprovementsFiles} )

# Get the files in the subdir Filters
file( GLOB OpenCL_FiltersFiles Filters/*.* )
source_group( "Filters" FILES ${OpenCL_FiltersFiles} )

# Get the files in the subdir Kernels
file( GLOB OpenCL_KernelsFiles Kernels/*.cl* )
source_group( "Kernels" FILES ${OpenCL_KernelsFiles} )

write_gpu_kernels( "${OpenCL_KernelsFiles}" OpenCL_KernelsFiles_SRC "Kernels Generated" )

# Set the files, so that they are shown in the editor
SET( OpenCL_Files
  ${OpenCL_CommonFiles}
  ${OpenCL_ITKimprovementsFiles}
  ${OpenCL_FiltersFiles}
  ${OpenCL_KernelsFiles}
  ${OpenCL_KernelsFiles_SRC} )

# Create the openCL library
add_library( elastix_opencl ${OpenCL_Files} )

# Link it against the necessary libraries.
target_link_libraries( elastix_opencl
  ${ITK_LIBRARIES}
  ${OPENCL_LIBRARIES} )

# Group in IDE's like Visual Studio
set_property( TARGET elastix_opencl PROPERTY FOLDER "libraries" )

