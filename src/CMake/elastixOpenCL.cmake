#
# \note This file is a copy of the ITK 4.2.0 CMake\itkOpenCL.cmake
# We added support for .clh file(OpenCL header kernels)
# It was modified by Denis P. Shamonin and Marius Staring.
# Division of Image Processing,
# Department of Radiology, Leiden, The Netherlands.
# Added functionality is described in the Insight Journal paper:
# http://hdl.handle.net/10380/3393
#

#-----------------------------------------------------------------------------
# OpenCL interface macros.
# opencl_source_file_to_string(<module>)
#
macro(opencl_source_file_to_string _source_file _result_cmake_var)
  file(STRINGS ${_source_file} FileStrings)
  foreach(SourceLine ${FileStrings})
    # replace all \ with \\ to make the c string constant work
    string(REGEX REPLACE "\\\\" "\\\\\\\\" TempSourceLine "${SourceLine}")
    # replace all " with \" to make the c string constant work
    string(REGEX REPLACE "\"" "\\\\\"" EscapedSourceLine "${TempSourceLine}")
    set(${_result_cmake_var} "${${_result_cmake_var}}\n\"${EscapedSourceLine}\\n\"")
  endforeach()
endmacro()

#-----------------------------------------------------------------------------
# OpenCL interface macros.
# write_opencl_kernel_to_file(<module>)
#
macro(write_opencl_kernel_to_file _opencl_file _src_var _group_name)
  #message(STATUS "write_opencl_kernel_to_file macro variables")
  #message(STATUS "VAR _opencl_file: ${_opencl_file}")
  #message(STATUS "VAR _src_var: ${_src_var}")
  #message(STATUS "VAR _group_name: ${_group_name}")

  # get file name and extension
  get_filename_component(OpenCLFileName ${_opencl_file} NAME_WE)
  get_filename_component(OpenCLFileExtension ${_opencl_file} EXT)

  # define kernel class, output filename for generated file and opencl file copy
  set(kernel_cxx_include "itk${OpenCLFileName}.h")
  set(kernel_cxx_class_name)
  set(output_file_generated)
  set(opencl_file_copy)

  if(${OpenCLFileExtension} STREQUAL ".clh")
    set(kernel_cxx_class_name ${OpenCLFileName}HeaderKernel)
    set(output_file_generated "${OpenCLFileName}HeaderKernel.cxx")
    set(opencl_file_copy "${output_file_generated}.clh")
  elseif(${OpenCLFileExtension} STREQUAL ".cl")
    set(kernel_cxx_class_name ${OpenCLFileName}Kernel)
    set(output_file_generated "${OpenCLFileName}Kernel.cxx")
    set(opencl_file_copy "${output_file_generated}.cl")
  endif()

  #message(STATUS "kernel_cxx_include   : ${kernel_cxx_include}")
  #message(STATUS "kernel_cxx_class_name: ${kernel_cxx_class_name}")
  #message(STATUS "output_file_generated: ${output_file_generated}")
  #message(STATUS "opencl_file_copy     : ${opencl_file_copy}")

  # convert file to string
  opencl_source_file_to_string(${_opencl_file} ${kernel_cxx_class_name}_SourceString)

  # add include on top
  set(${kernel_cxx_class_name}_KernelString
    "#include \"${kernel_cxx_include}\"\n\n")

  # add namespace itk
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}namespace itk\n")

  # add const char* ${_kernel_cxx_class_name}::GetOpenCLSource() here
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}{\n\n")
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}const char* ${kernel_cxx_class_name}::GetOpenCLSource()\n")
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}{\n")

  # add converted string
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}  return ${${kernel_cxx_class_name}_SourceString};\n")
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}}\n\n")

  # add closing bracket to namespace itk
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}} // namespace itk\n")

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated}
    "${${kernel_cxx_class_name}_KernelString}")

  configure_file(${_opencl_file} ${CMAKE_CURRENT_BINARY_DIR}/${opencl_file_copy} COPYONLY)

  add_custom_target(${kernel_cxx_class_name}_Target
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated}
      ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
      ${CMAKE_CURRENT_BINARY_DIR}/${opencl_file_copy}
  )

  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated} PROPERTIES GENERATED ON)
  set(${_src_var} ${${_src_var}} ${output_file_generated})

  set_property(TARGET ${kernel_cxx_class_name}_Target PROPERTY FOLDER ${_group_name})
endmacro()

#-----------------------------------------------------------------------------
# OpenCL interface macros.
# write_opencl_kernels_to_file(<module>)
#
macro(write_opencl_kernels_to_file _opencl_files _merge_to _src_var _group_name)
  #message(STATUS "write_opencl_kernels_to_file macro variables")
  #message(STATUS "VAR _opencl_files: ${_opencl_files}")
  #message(STATUS "VAR _merge_to: ${_merge_to}")
  #message(STATUS "VAR _src_var: ${_src_var}")
  #message(STATUS "VAR _group_name: ${_group_name}")

  set(kernel_cxx_class_name ${_merge_to}Kernel)
  #message(STATUS "kernel_cxx_class_name: ${kernel_cxx_class_name}")

  set(output_file_generated "${kernel_cxx_class_name}.cxx")

  # add include on top
  set(${kernel_cxx_class_name}_KernelString
    "#include \"itk${_merge_to}.h\"\n\n")

  # add namespace itk
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}namespace itk\n{\n")

  foreach(opencl_file ${_opencl_files})
    # get file name and extension
    get_filename_component(OpenCLFileName ${opencl_file} NAME_WE)
    get_filename_component(OpenCLFileExtension ${opencl_file} EXT)

    # get the file name
    get_filename_component(OpenCLFileName ${opencl_file} NAME_WE)

    # convert file to string
    opencl_source_file_to_string(${opencl_file} ${OpenCLFileName}_SourceString)

    # add const char* ${kernel_cxx_class_name}::GetOpenCLSource() here
    set(${OpenCLFileName}_KernelString
      "${${OpenCLFileName}_KernelString}const char* ${OpenCLFileName}Kernel::GetOpenCLSource()\n")
    set(${OpenCLFileName}_KernelString
      "${${OpenCLFileName}_KernelString}{\n")

    # add converted string
    set(${OpenCLFileName}_KernelString
      "${${OpenCLFileName}_KernelString}  return ${${OpenCLFileName}_SourceString};\n")
    set(${OpenCLFileName}_KernelString
      "${${OpenCLFileName}_KernelString}}\n\n")

    # append to ${kernel_cxx_class_name}_KernelString
    set(${kernel_cxx_class_name}_KernelString
      "${${kernel_cxx_class_name}_KernelString} ${${OpenCLFileName}_KernelString}")
  endforeach()

  # add closing bracket to namespace itk
  set(${kernel_cxx_class_name}_KernelString
    "${${kernel_cxx_class_name}_KernelString}} // namespace itk\n")

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated}
    "${${kernel_cxx_class_name}_KernelString}")

  # make sure that if we modify original OpenCL file from ${_opencl_files}, then kernel string has to be recreated
  set(opencl_depend_files "")
  list(APPEND opencl_depend_files "${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated}")
  list(APPEND opencl_depend_files "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")

  foreach(opencl_file ${_opencl_files})
    get_filename_component(OpenCLFileName ${opencl_file} NAME_WE)
    get_filename_component(OpenCLFileExtension ${opencl_file} EXT)

    set(opencl_file_copy)

    if(${OpenCLFileExtension} STREQUAL ".clh")
      set(opencl_file_copy "${OpenCLFileName}HeaderKernel.cxx.clh")
    elseif(${OpenCLFileExtension} STREQUAL ".cl")
      set(opencl_file_copy "${OpenCLFileName}Kernel.cxx.cl")
    endif()

    configure_file(${opencl_file} ${CMAKE_CURRENT_BINARY_DIR}/${opencl_file_copy} COPYONLY)
    list(APPEND opencl_depend_files "${CMAKE_CURRENT_BINARY_DIR}/${opencl_file_copy}")
  endforeach()

  #message(STATUS "VAR opencl_depend_files: ${opencl_depend_files}")

  add_custom_target(${kernel_cxx_class_name}_Target
    DEPENDS
      ${opencl_depend_files}
  )

  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${output_file_generated} PROPERTIES GENERATED ON)
  set(${_src_var} ${${_src_var}} ${output_file_generated})
  set_property(TARGET ${kernel_cxx_class_name}_Target PROPERTY FOLDER ${_group_name})
endmacro()

#-----------------------------------------------------------------------------
# OpenCL interface macros.
# write_opencl_kernels(<module>)
#
macro(write_opencl_kernels _opencl_kernels _opencl_src _group_name _merge_kernels_to_one_file)
  #message(STATUS "write_opencl_kernels macro variables")
  #message(STATUS "VAR _opencl_kernels: ${_opencl_kernels}")
  #message(STATUS "VAR _opencl_src: ${_opencl_src}")
  #message(STATUS "VAR _group_name: ${_group_name}")
  #message(STATUS "VAR _merge_kernels_to_one_file: ${_merge_kernels_to_one_file}")

  if(NOT ${_merge_kernels_to_one_file})
    foreach(opencl_kernel ${_opencl_kernels})
      write_opencl_kernel_to_file(${opencl_kernel} ${_opencl_src} ${_group_name})
    endforeach()
  else()
    set(merge_to ${ARGV4})
    write_opencl_kernels_to_file("${_opencl_kernels}" ${merge_to} ${_opencl_src} ${_group_name})
  endif()
endmacro()
