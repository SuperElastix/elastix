# This file is a copy of the ITK4.2.0 CMake\itkOpenCL.cmake
# We added support for .clh file (OpenCL header kernels)
# TODO: This file has to be moved in CMake directory

if( ELASTIX_USE_OPENCL )

  macro( sourcefile_to_string SOURCE_FILE RESULT_CMAKE_VAR )
    file( STRINGS ${SOURCE_FILE} FileStrings )
    foreach( SourceLine ${FileStrings} )
      # replace all \ with \\ to make the c string constant work
      string(REGEX REPLACE "\\\\" "\\\\\\\\" TempSourceLine "${SourceLine}")
      # replace all " with \" to make the c string constant work
      string(REGEX REPLACE "\"" "\\\\\"" EscapedSourceLine "${TempSourceLine}")
      set(${RESULT_CMAKE_VAR} "${${RESULT_CMAKE_VAR}}\n\"${EscapedSourceLine}\\n\"")
    endforeach()
  endmacro()

  macro( write_gpu_kernel_to_file OPENCL_FILE GPUFILTER_NAME GPUFILTER_KERNELNAME OUTPUT_FILE SRC_VAR GROUP_NAME )
    sourcefile_to_string( ${OPENCL_FILE} ${GPUFILTER_KERNELNAME}_SourceString )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "#include \"itk${GPUFILTER_NAME}.h\"\n\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}namespace itk\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}{\n\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}const char* ${GPUFILTER_KERNELNAME}::GetOpenCLSource()\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}{\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}  return ${${GPUFILTER_KERNELNAME}_SourceString};\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}}\n\n" )
    set( ${GPUFILTER_KERNELNAME}_KernelString
        "${${GPUFILTER_KERNELNAME}_KernelString}}\n" )

    file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
        "${${GPUFILTER_KERNELNAME}_KernelString}" )

    add_custom_target( ${GPUFILTER_KERNELNAME}_Target SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} )
    add_dependencies( ${GPUFILTER_KERNELNAME}_Target ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} )
    add_dependencies( ${GPUFILTER_KERNELNAME}_Target ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt )

    # make sure that if we modify original OpenCL file ${OPENCL_FILE}, then kernel string has to be recreated
    configure_file( ${OPENCL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}.cl COPYONLY )
    add_dependencies( ${GPUFILTER_KERNELNAME}_Target ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}.cl )

    set_source_files_properties( ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} PROPERTIES GENERATED ON )
    set( ${SRC_VAR} ${${SRC_VAR}} ${OUTPUT_FILE} )

    set_property( TARGET ${GPUFILTER_KERNELNAME}_Target PROPERTY FOLDER ${GROUP_NAME} )

  endmacro()

  macro( write_gpu_kernels GPUKernels GPU_SRC GROUP_NAME )
    foreach( GPUKernel ${GPUKernels} )
      get_filename_component( KernelFileName ${GPUKernel} NAME_WE )
      get_filename_component( KernelFileExtension ${GPUKernel} EXT )
      if( ${KernelFileExtension} STREQUAL ".clh" )
        write_gpu_kernel_to_file( ${GPUKernel} ${KernelFileName} ${KernelFileName}HeaderKernel
          "${KernelFileName}HeaderKernel.cxx" ${GPU_SRC} ${GROUP_NAME} )
      elseif( ${KernelFileExtension} STREQUAL ".cl" )
        write_gpu_kernel_to_file( ${GPUKernel} ${KernelFileName} ${KernelFileName}Kernel
          "${KernelFileName}Kernel.cxx" ${GPU_SRC} ${GROUP_NAME} )
      endif()
    endforeach()
  endmacro()

endif()

