
project(elxOpenCL)

# Define directory to OpenCL Debug kernels
set(OPENCL_KERNELS_DEBUG_PATH
  ${elastix_BINARY_DIR}
  CACHE INTERNAL "Path to itkOpenCLKernels.h")

set(OPENCL_KERNELS_DEBUG_DIR
  ${OPENCL_KERNELS_DEBUG_PATH}/DebugOpenCL
  CACHE INTERNAL "Directory to OpenCL debug kernels")

file(MAKE_DIRECTORY ${OPENCL_KERNELS_DEBUG_DIR})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ITKimprovements/itkOpenCLKernels.h.in
  ${OPENCL_KERNELS_DEBUG_PATH}/itkOpenCLKernels.h
  @ONLY
)

# Get the files in the main directory
file(GLOB OpenCL_CommonFiles *.*)
list(APPEND OpenCL_CommonFiles ${OPENCL_KERNELS_DEBUG_PATH}/itkOpenCLKernels.h)

source_group("Common" FILES ${OpenCL_CommonFiles})

# Get the files in the subdir Factories
file(GLOB OpenCL_FactoriesFiles Factories/*.*)
source_group("Factories" FILES ${OpenCL_FactoriesFiles})

# Get the files in the subdir ITKimprovements
file(GLOB OpenCL_ITKimprovementsFiles ITKimprovements/*.*)
source_group("ITKimprovements" FILES ${OpenCL_ITKimprovementsFiles})

# Get the files in the subdir Filters
file(GLOB OpenCL_FiltersFiles Filters/*.*)
source_group("Filters" FILES ${OpenCL_FiltersFiles})

# Get the files in the subdir Kernels
file(GLOB OpenCL_KernelsFiles Kernels/*.cl)
source_group("Kernels" FILES ${OpenCL_KernelsFiles})

write_opencl_kernels("${OpenCL_KernelsFiles}" OpenCL_KernelsFiles_SRC "OpenCL generated kernels" OFF)

# Set the files, so that they are shown in the editor
set(OpenCL_Files
  ${OpenCL_CommonFiles}
  ${OpenCL_ITKimprovementsFiles}
  ${OpenCL_FactoriesFiles}
  ${OpenCL_FiltersFiles}
  ${OpenCL_KernelsFiles}
  ${OpenCL_KernelsFiles_SRC}
)

# Create the openCL library
add_library(elxOpenCL STATIC ${OpenCL_Files})

# Link it against the necessary libraries.
target_link_libraries(elxOpenCL
  ${ITK_LIBRARIES}
  ${OPENCL_LIBRARIES}
)

target_include_directories(elxOpenCL PUBLIC ${OPENCL_INCLUDE_DIRS})
target_compile_definitions(elxOpenCL PUBLIC ELASTIX_USE_OPENCL)

# The key-word BEFORE is important here, because the elastix distribution
# contains some files with the same name as in the ITK distribution.
# Some bugs were fixed and features added. When these are contributed back
# to the ITK the BEFORE keyword is not needed anymore.
target_include_directories(elxOpenCL BEFORE PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Factories>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ITKimprovements>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Filters>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Kernels>
  $<BUILD_INTERFACE:${OPENCL_KERNELS_DEBUG_PATH}>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/OpenCL>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/OpenCL/Factories>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/OpenCL/ITKimprovements>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/OpenCL/Filters>
  $<INSTALL_INTERFACE:${ELASTIX_INCLUDE_DIR}/Common/OpenCL/Kernels>
)

# Group in IDE's like Visual Studio
set_property(TARGET elxOpenCL PROPERTY FOLDER "libraries")

if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(TARGETS elxOpenCL
    ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
    LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
    RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
    COMPONENT Development)
endif()

elastix_export_target(elxOpenCL)
