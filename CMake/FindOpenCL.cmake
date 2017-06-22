# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system.
# FindOpenCL.cmake supports following implementations of the OpenCL:
#   - Intel OpenCL SDK (Intel SDK for OpenCL)
#   - AMD OpenCL SDK (AMD APP SDK)
#   - NVidia OpenCL SDK (part of the NVIDIA CUDA Toolkit).
# It also supports multiple OpenCL implementation and switching between them.
#
# By default FindOpenCL.cmake first tries to locate NVidia OpenCL SDK or AMD OpenCL SDK.
# If they don't exist, search for Intel OpenCL SDK is performed. If system has multiple
# OpenCL SDK installed (for example NVidia OpenCL SDK and Intel OpenCL SDK) FindOpenCL.cmake
# will locate them and enable or disable the following flags:
#   OPENCL_INTEL_FOUND, OPENCL_NVIDIA_FOUND, OPENCL_AMD_FOUND (see documentation below).
# User may manually control preferred OpenCL SDK by enabling or disabling following flags:
#   OPENCL_USE_INTEL_SDK, OPENCL_USE_NVIDIA_SDK, OPENCL_USE_AMD_SDK (see documentation below).
# For Intel and AMD OpenCL SDK's switching between CPU and GPU implementations controls by:
#   OPENCL_USE_INTEL_SDK_GPU_CPU, OPENCL_USE_AMD_SDK_GPU_CPU (see documentation below).
# In addition OpenCL profiling controls by OPENCL_PROFILING flag.
# Various OpenCL math, optimization, warnings suppression and opencl c version
# could be adjusted as well (see documentation below).
#
# Once done this will define:
#  OPENCL_FOUND         - system has OpenCL.
#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory. Path to OpenCL header file 'CL/cl.h'
#  OPENCL_LIBRARIES     - link these to use OpenCL. Path to OpenCL library 'OpenCL.lib'
#
# Other variables defined by this module listed below.
#  OPENCL_SDK_STRING - Hold selected OpenCL SDK. For example could be used for setting
#  project name of your project (see example below).
#   set( project_name_combined "myproject-${OPENCL_SDK_STRING}" CACHE INTERNAL "internal project name" )
#   string( REPLACE " " "-" project_name_combined ${project_name_combined} )
#   set( myproject_project_name ${project_name_combined} CACHE string "project name" FORCE )
#   project(${myproject_project_name})
#
# Intel OpenCL SDK variables:
#  OPENCL_INTEL_FOUND           - true if Intel OpenCL SDK has been found on this system.
#  OPENCL_INTEL_INCLUDE_DIR     - path to Intel OpenCL header file 'CL/cl.h'.
#  OPENCL_INTEL_LIBRARY         - path to Intel OpenCL library 'OpenCL.lib'.
#  OPENCL_USE_INTEL_SDK         - set Intel OpenCL SDK as selected.
#  OPENCL_USE_INTEL_SDK_GPU_CPU - use Intel GPU or CPU implementation of the OpenCL.
#                                 On use GPU, Off use CPU.
#
# NVidia OpenCL SDK variables:
#  OPENCL_NVIDIA_FOUND          - true if NVidia OpenCL SDK has been found on this system.
#  OPENCL_NVIDIA_INCLUDE_DIR    - path to NVidia OpenCL header file 'CL/cl.h'.
#  OPENCL_NVIDIA_LIBRARY        - path to NVidia OpenCL library 'OpenCL.lib'
#  OPENCL_USE_NVIDIA_SDK        - set NVidia OpenCL SDK as selected.
#
# AMD OpenCL SDK variables:
#  OPENCL_AMD_FOUND             - true if AMD OpenCL SDK has been found on this system.
#  OPENCL_AMD_INCLUDE_DIR       - path to AMD OpenCL header file 'CL/cl.h'
#  OPENCL_AMD_LIBRARY           - path to AMD OpenCL library 'OpenCL.lib'
#  OPENCL_USE_AMD_SDK           - set AMD OpenCL SDK as selected.
#  OPENCL_USE_AMD_SDK_GPU_CPU   - use AMD GPU or CPU implementation of the OpenCL.
#                                 On use GPU, Off use CPU.
#
# OpenCL profiling options:
#  OPENCL_PROFILING - Enable OpenCL profiling with CL_QUEUE_PROFILING_ENABLE.
#   Event objects can be used to capture profiling information that measure execution time of a command.
#
# OpenCL math intrinsics options:
#  OPENCL_MATH_SINGLE_PRECISION_CONSTANT - Treat double precision floating-point constant
#   as single precision constant.
#  OPENCL_MATH_DENORMS_ARE_ZERO CACHE BOOL - This option controls how single precision and double
#   precision denormalized numbers are handled.
#  OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT - This option allows an application to specify
#   that single precision floating-point divide (x/y and 1/x) and sqrt used
#   in the program source are correctly rounded.
#
# OpenCL optimization options:
#  OPENCL_OPTIMIZATION_OPT_DISABLE - This option disables all optimizations.
#   The default is optimizations are enabled.
#  OPENCL_OPTIMIZATION_MAD_ENABLE - Allow a * b + c to be replaced by a mad.
#   The mad computes a * b + c with reduced accuracy.
#  OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS - Allow optimizations for floating-point arithmetic that
#   ignore the signedness of zero.
#  OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS - Allow optimizations for floating-point arithmetic.
#  OPENCL_OPTIMIZATION_FINITE_MATH_ONLY - Allow optimizations for floating-point arithmetic that
#   assume that arguments and results are not NaNs or +-infinity.
#  OPENCL_OPTIMIZATION_FAST_RELAXED_MATH - Sets the optimization options
#   -cl-finite-math-only and -cl-unsafe-math-optimizations.
#  OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE - This requires that the global work-size be a
#   multiple of the work-group size specified to clEnqueueNDRangeKernel.
#
# OpenCL options to request or suppress warnings:
#  OPENCL_WARNINGS_DISABLE - This option inhibit all warning messages.
#  OPENCL_WARNINGS_AS_ERRORS - This option make all warnings into errors.
#
# OpenCL options controlling the opencl c version:
#  OPENCL_C_VERSION_1_1 - This option determine the OpenCL C language version to use.
#   Support all OpenCL C programs that use the OpenCL C language 1.1 specification.
#  OPENCL_C_VERSION_1_2 - This option determine the OpenCL C language version to use.
#   Support all OpenCL C programs that use the OpenCL C language 1.2 specification.
#  OPENCL_C_VERSION_2_0 - This option determine the OpenCL C language version to use.
#   Support all OpenCL C programs that use the OpenCL C language 2.0 specification.
#
#=============================================================================
# \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
# Department of Radiology, Leiden, The Netherlands
#
# \note This work was funded by the Netherlands Organisation for
# Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
#=============================================================================
# (To distribute this file outside of elastix, substitute the full author
# text for the above reference.)

find_package( PackageHandleStandardArgs )

#=============================================================================
# Macros for FindOpenCL.cmake

# macro to set FindOpenCL.cmake initial state
macro( opencl_init )
  # Define variable OPENCL_DEFAULT_SDK_SELECTED
  if( NOT DEFINED OPENCL_DEFAULT_SDK_SELECTED )
    set( OPENCL_DEFAULT_SDK_SELECTED FALSE CACHE INTERNAL "OpenCL SDK has been selected by default." )
  endif()

  # Define OPENCL_INCLUDE_DIRS
  if( NOT DEFINED OPENCL_INCLUDE_DIRS )
    set( OPENCL_INCLUDE_DIRS OPENCL_INCLUDE_DIRS-NOTFOUND CACHE PATH "Path to OpenCL header file 'CL/cl.h'." )
  else()
    if( NOT OPENCL_INCLUDE_DIRS MATCHES NOTFOUND )
      if( NOT EXISTS "${OPENCL_INCLUDE_DIRS}/CL/cl.h" )
        message( WARNING "OpenCL header 'cl.h' has not been found at '${OPENCL_INCLUDE_DIRS}/CL'" )
        set( OPENCL_INCLUDE_DIRS OPENCL_INCLUDE_DIRS-NOTFOUND CACHE PATH "Path to OpenCL header file 'CL/cl.h'." FORCE )
      endif()
    endif()
  endif()

  # Define OPENCL_LIBRARIES
  if( NOT DEFINED OPENCL_LIBRARIES )
    set( OPENCL_LIBRARIES OPENCL_LIBRARIES-NOTFOUND CACHE PATH "Path to OpenCL library 'OpenCL.lib'." )
  else()
    if( NOT OPENCL_LIBRARIES MATCHES NOTFOUND )
      if( NOT EXISTS ${OPENCL_LIBRARIES} )
        message( WARNING "OpenCL library 'OpenCL.lib' has not been found at '${OPENCL_LIBRARIES}'" )
        set( OPENCL_LIBRARIES OPENCL_LIBRARIES-NOTFOUND CACHE PATH "Path to OpenCL library 'OpenCL.lib'." FORCE )
      endif()
    endif()
  endif()

  # Define OpenCL header name
  if( WIN32 OR UNIX )
    set( OPENCL_HEADER_NAME "CL/cl.h" CACHE INTERNAL "OpenCL header name CL/cl.h (Win and Unix) and OpenCL/cl.h (Apple)." )
  elseif( APPLE )
    set( OPENCL_HEADER_NAME "OpenCL/cl.h" CACHE INTERNAL "OpenCL header name CL/cl.h (Win and Unix) and OpenCL/cl.h (Apple)." )
  endif()
endmacro()

# macro to find Intel OpenCL SDK
macro( opencl_find_intel )
  # Set find result to false
  set( OPENCL_INTEL_FOUND FALSE CACHE INTERNAL "Intel OpenCL SDK has not been found on this system." )

  # Find cl.h in Intel OpenCL SDK
  find_path( OPENCL_INTEL_INCLUDE_DIR ${OPENCL_HEADER_NAME}
    HINTS
      $ENV{INTELOCLSDKROOT}
    PATH_SUFFIXES include
    PATHS /usr
    NO_DEFAULT_PATH
  )

  # Make it internal
  set( OPENCL_INTEL_INCLUDE_DIR ${OPENCL_INTEL_INCLUDE_DIR} CACHE INTERNAL "Intel OpenCL include directory." )

  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    find_library( OPENCL_INTEL_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{INTELOCLSDKROOT}
      PATH_SUFFIXES lib/x64
      PATHS /usr/lib
      NO_DEFAULT_PATH
    )
  else()
    find_library( OPENCL_INTEL_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{INTELOCLSDKROOT}
      PATH_SUFFIXES lib/x86
      PATHS /usr/lib
      NO_DEFAULT_PATH
    )
  endif()

  # Make it internal
  set( OPENCL_INTEL_LIBRARY ${OPENCL_INTEL_LIBRARY} CACHE INTERNAL "Intel OpenCL library." )

  if( NOT (${OPENCL_INTEL_INCLUDE_DIR} STREQUAL OPENCL_INTEL_INCLUDE_DIR-NOTFOUND)
      AND NOT (${OPENCL_INTEL_LIBRARY} STREQUAL OPENCL_INTEL_LIBRARY-NOTFOUND) )
    set( OPENCL_INTEL_FOUND TRUE CACHE INTERNAL "Intel OpenCL SDK has not been found on this system." )
  endif()
endmacro()

# macro to find NVidia OpenCL SDK
macro( opencl_find_nvidia )
  # Set find result to false
  set( OPENCL_NVIDIA_FOUND FALSE CACHE INTERNAL "NVidia OpenCL SDK has not been found on this system." )

  # Find cl.h in NVidia OpenCL SDK
  find_path( OPENCL_NVIDIA_INCLUDE_DIR ${OPENCL_HEADER_NAME}
    HINTS
      $ENV{CUDA_PATH}
    PATH_SUFFIXES include
    PATHS /usr/local/cuda
    NO_DEFAULT_PATH
  )

  # Make it internal
  set( OPENCL_NVIDIA_INCLUDE_DIR ${OPENCL_NVIDIA_INCLUDE_DIR} CACHE INTERNAL "NVidia OpenCL include directory." )

  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    find_library( OPENCL_NVIDIA_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{CUDA_PATH}
      PATH_SUFFIXES lib/x64
      PATHS /usr/local/cuda/lib64
      NO_DEFAULT_PATH
    )
  else()
    find_library( OPENCL_NVIDIA_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{CUDA_PATH}
      PATH_SUFFIXES lib/Win32
      PATHS /usr/local/cuda/lib
      NO_DEFAULT_PATH
    )
  endif()

  # Make it internal
  set( OPENCL_NVIDIA_LIBRARY ${OPENCL_NVIDIA_LIBRARY} CACHE INTERNAL "NVidia OpenCL library." )

  if( NOT (${OPENCL_NVIDIA_INCLUDE_DIR} STREQUAL OPENCL_NVIDIA_INCLUDE_DIR-NOTFOUND)
      AND NOT (${OPENCL_NVIDIA_LIBRARY} STREQUAL OPENCL_NVIDIA_LIBRARY-NOTFOUND))
    set( OPENCL_NVIDIA_FOUND TRUE CACHE INTERNAL "NVidia OpenCL SDK has been found on this system." )
  endif()
endmacro()

# macro to find AMD OpenCL SDK
macro( opencl_find_amd )
  # Set find result to false
  set( OPENCL_AMD_FOUND FALSE CACHE INTERNAL "AMD OpenCL SDK has not been found on this system." )

  # Find cl.h in AMD OpenCL SDK
  find_path( OPENCL_AMD_INCLUDE_DIR ${OPENCL_HEADER_NAME}
    HINTS
      $ENV{AMDAPPSDKROOT}
    PATH_SUFFIXES include
    PATHS /opt/AMDAPP
    NO_DEFAULT_PATH
  )

  # Make it internal
  set( OPENCL_AMD_INCLUDE_DIR ${OPENCL_AMD_INCLUDE_DIR} CACHE INTERNAL "AMD OpenCL include directory." )

  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    find_library( OPENCL_AMD_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{AMDAPPSDKROOT}
      PATH_SUFFIXES lib/x86_64
      PATHS /opt/AMDAPP/lib
      NO_DEFAULT_PATH
    )
  else()
    find_library( OPENCL_AMD_LIBRARY
      NAMES OpenCL
      HINTS
        $ENV{AMDAPPSDKROOT}
      PATH_SUFFIXES lib/x86
      PATHS /opt/AMDAPP/lib
      NO_DEFAULT_PATH
    )
  endif()

  # Make it internal
  set( OPENCL_AMD_LIBRARY ${OPENCL_AMD_LIBRARY} CACHE INTERNAL "AMD OpenCL library." )

  if( NOT (${OPENCL_AMD_INCLUDE_DIR} STREQUAL OPENCL_AMD_INCLUDE_DIR-NOTFOUND)
      AND NOT (${OPENCL_AMD_LIBRARY} STREQUAL OPENCL_AMD_LIBRARY-NOTFOUND))
    set( OPENCL_AMD_FOUND TRUE CACHE INTERNAL "AMD OpenCL SDK has not been found on this system." )
  endif()
endmacro()

# macro opencl_define_avaliable_sdk
macro( opencl_define_avaliable_sdk )
  # Define OPENCL_USE_INTEL_SDK
  if( NOT DEFINED OPENCL_USE_INTEL_SDK AND OPENCL_INTEL_FOUND )
    if( NOT OPENCL_USE_NVIDIA_SDK AND NOT OPENCL_USE_AMD_SDK )
      set( OPENCL_USE_INTEL_SDK FALSE CACHE BOOL
        "Use Intel implementation of the OpenCL." )
      set( OPENCL_USE_INTEL_SDK_GPU_CPU TRUE CACHE BOOL
        "Use Intel GPU or CPU implementation of the OpenCL. On use GPU, Off use CPU." )
    endif()
  elseif( DEFINED OPENCL_USE_INTEL_SDK )
    if( OPENCL_USE_NVIDIA_SDK OR OPENCL_USE_AMD_SDK )
      unset( OPENCL_USE_INTEL_SDK CACHE )
      unset( OPENCL_USE_INTEL_SDK_GPU_CPU CACHE )
    endif()
  endif()

  # Define OPENCL_USE_NVIDIA_SDK
  if( NOT DEFINED OPENCL_USE_NVIDIA_SDK AND OPENCL_NVIDIA_FOUND )
    if( NOT OPENCL_USE_INTEL_SDK AND NOT OPENCL_USE_AMD_SDK )
      set( OPENCL_USE_NVIDIA_SDK FALSE CACHE BOOL
        "Use NVidia implementation of the OpenCL." )
    endif()
  elseif( DEFINED OPENCL_USE_NVIDIA_SDK )
    if( OPENCL_USE_INTEL_SDK OR OPENCL_USE_AMD_SDK )
      unset( OPENCL_USE_NVIDIA_SDK CACHE )
    endif()
  endif()

  # Define OPENCL_USE_AMD_SDK
  if( NOT DEFINED OPENCL_USE_AMD_SDK AND OPENCL_AMD_FOUND )
    if( NOT OPENCL_USE_INTEL_SDK AND NOT OPENCL_USE_NVIDIA_SDK )
      set( OPENCL_USE_AMD_SDK FALSE CACHE BOOL
        "Use AMD implementation of the OpenCL." )
      set( OPENCL_USE_AMD_SDK_GPU_CPU TRUE CACHE BOOL
        "Use AMD GPU or CPU implementation of the OpenCL. On use GPU, Off use CPU." )
    endif()
  elseif( DEFINED OPENCL_USE_AMD_SDK )
    if( OPENCL_USE_INTEL_SDK OR OPENCL_USE_NVIDIA_SDK )
      unset( OPENCL_USE_AMD_SDK CACHE )
      unset( OPENCL_USE_AMD_SDK_GPU_CPU CACHE )
    endif()
  endif()
endmacro()

# macro opencl_select_sdk
macro( opencl_select_sdk )
  if( OPENCL_USE_INTEL_SDK )
    set( OPENCL_INCLUDE_DIRS ${OPENCL_INTEL_INCLUDE_DIR} CACHE PATH "Path to Intel OpenCL header file 'CL/cl.h'." FORCE )
    set( OPENCL_LIBRARIES ${OPENCL_INTEL_LIBRARY} CACHE PATH "Path to Intel OpenCL library 'OpenCL.lib'." FORCE )

    # Set CMAKE_CXX_FLAGS
    if( OPENCL_USE_INTEL_SDK_GPU_CPU )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_INTEL_GPU" )
    else()
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_INTEL_CPU" )
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_INTEL" )
    set( OPENCL_SDK_STRING "Intel OpenCL" CACHE INTERNAL "Selected OpenCL SDK" )
  elseif( OPENCL_USE_NVIDIA_SDK )
    set( OPENCL_INCLUDE_DIRS ${OPENCL_NVIDIA_INCLUDE_DIR} CACHE PATH "Path to NVidia OpenCL header file 'CL/cl.h'." FORCE )
    set( OPENCL_LIBRARIES ${OPENCL_NVIDIA_LIBRARY} CACHE PATH "Path to NVidia OpenCL library 'OpenCL.lib'." FORCE )

    # Set CMAKE_CXX_FLAGS
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_NVIDIA" )
    set( OPENCL_SDK_STRING "NVidia OpenCL" CACHE INTERNAL "Selected OpenCL SDK" )
  elseif( OPENCL_USE_AMD_SDK )
    set( OPENCL_INCLUDE_DIRS ${OPENCL_AMD_INCLUDE_DIR} CACHE PATH "Path to AMD OpenCL header file 'CL/cl.h'." FORCE )
    set( OPENCL_LIBRARIES ${OPENCL_AMD_LIBRARY} CACHE PATH "Path to AMD OpenCL library 'OpenCL.lib'." FORCE )

    # Set CMAKE_CXX_FLAGS
    if( OPENCL_USE_AMD_SDK_GPU_CPU )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_AMD_GPU" )
    else()
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_AMD_CPU" )
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_USE_AMD -DATI_OS_WIN" )
    set( OPENCL_SDK_STRING "AMD OpenCL" CACHE INTERNAL "Selected OpenCL SDK" )
  else()
    set( OPENCL_SDK_STRING "OpenCL not found" CACHE INTERNAL "Selected OpenCL SDK" )
    #message( WARNING "Please select OpenCL platform" )
  endif()
endmacro()

# macro opencl_select_default_sdk
macro( opencl_select_default_sdk )
  if( NOT OPENCL_DEFAULT_SDK_SELECTED
      AND NOT OPENCL_USE_INTEL_SDK
      AND NOT OPENCL_USE_NVIDIA_SDK
      AND NOT OPENCL_USE_AMD_SDK )

    # If both NVidia SDK and AMD SDK exist on the system,
    # then let user to specify which platform to use
    if( OPENCL_NVIDIA_FOUND AND OPENCL_AMD_FOUND )
      message( SEND_ERROR "Both NVidia OpenCL SDK and AMD OpenCL SDK exists on this system. Please select the OpenCL platform." )
    # If NVidia SDK has been found without AMD SDK
    elseif( OPENCL_NVIDIA_FOUND AND NOT OPENCL_AMD_FOUND )
      set( OPENCL_USE_NVIDIA_SDK TRUE CACHE BOOL
        "Use NVidia implementation of the OpenCL." FORCE )
      opencl_select_sdk()
    # If AMD SDK has been found without NVidia SDK
    elseif( OPENCL_AMD_FOUND AND NOT OPENCL_NVIDIA_FOUND )
      set( OPENCL_USE_AMD_SDK TRUE CACHE BOOL
        "Use AMD implementation of the OpenCL." FORCE )
      opencl_select_sdk()
    # If only Intel SDK has been found without NVidia SDK or AMD SDK
    elseif( OPENCL_INTEL_FOUND AND (NOT OPENCL_NVIDIA_FOUND OR NOT OPENCL_AMD_FOUND) )
      set( OPENCL_USE_INTEL_SDK TRUE CACHE BOOL
        "Use Intel implementation of the OpenCL." FORCE )
      opencl_select_sdk()
    endif()

    # Set the selected flag
    set( OPENCL_DEFAULT_SDK_SELECTED TRUE CACHE INTERNAL "OpenCL SDK has been selected by default." )
  endif()
endmacro()

# macro opencl_define_options
macro( opencl_define_options )
  if( OPENCL_USE_INTEL_SDK OR OPENCL_USE_NVIDIA_SDK OR OPENCL_USE_AMD_SDK )
    # Define profiling variable for ITK4OpenCL
    set( OPENCL_PROFILING OFF CACHE BOOL
      "Enable OpenCL profiling with CL_QUEUE_PROFILING_ENABLE. Event objects can be used to capture profiling information that measure execution time of a command." )

    mark_as_advanced( OPENCL_PROFILING )

    # OpenCL Math Intrinsics Options
    set( OPENCL_MATH_SINGLE_PRECISION_CONSTANT CACHE BOOL
      "Treat double precision floating-point constant as single precision constant." )
    set( OPENCL_MATH_DENORMS_ARE_ZERO CACHE BOOL
      "This option controls how single precision and double precision denormalized numbers are handled." )
    set( OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT CACHE BOOL
      "This option allows an application to specify that single precision floating-point divide (x/y and 1/x) and sqrt used in the program source are correctly rounded." )

    mark_as_advanced( OPENCL_MATH_SINGLE_PRECISION_CONSTANT )
    mark_as_advanced( OPENCL_MATH_DENORMS_ARE_ZERO )
    mark_as_advanced( OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT )

    # OpenCL Optimization Options
    set( OPENCL_OPTIMIZATION_OPT_DISABLE CACHE BOOL
      "This option disables all optimizations. The default is optimizations are enabled." )
    set( OPENCL_OPTIMIZATION_MAD_ENABLE CACHE BOOL
      "Allow a * b + c to be replaced by a mad. The mad computes a * b + c with reduced accuracy." )
    set( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS CACHE BOOL
      "Allow optimizations for floating-point arithmetic that ignore the signedness of zero." )
    set( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE BOOL
      "Allow optimizations for floating-point arithmetic." )
    set( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE BOOL
      "Allow optimizations for floating-point arithmetic that assume that arguments and results are not NaNs or +-infinity." )
    set( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH CACHE BOOL
      "Sets the optimization options -cl-finite-math-only and -cl-unsafe-math-optimizations." )
    set( OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE CACHE BOOL
      "This requires that the global work-size be a multiple of the work-group size specified to clEnqueueNDRangeKernel." )

    mark_as_advanced( OPENCL_OPTIMIZATION_OPT_DISABLE )
    mark_as_advanced( OPENCL_OPTIMIZATION_MAD_ENABLE )
    mark_as_advanced( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS )
    mark_as_advanced( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS )
    mark_as_advanced( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY )
    mark_as_advanced( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH )
    mark_as_advanced( OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE )

    # OpenCL Options to Request or Suppress Warnings
    set( OPENCL_WARNINGS_DISABLE CACHE BOOL
      "This option inhibit all warning messages." )
    set( OPENCL_WARNINGS_AS_ERRORS CACHE BOOL
      "This option make all warnings into errors." )

    mark_as_advanced( OPENCL_WARNINGS_DISABLE )
    mark_as_advanced( OPENCL_WARNINGS_AS_ERRORS )

    # OpenCL Options Controlling the OpenCL C Version
    set( OPENCL_C_VERSION_1_1 CACHE BOOL
      "This option determine the OpenCL C language version to use. Support all OpenCL C programs that use the OpenCL C language 1.1 specification." )
    set( OPENCL_C_VERSION_1_2 CACHE BOOL
      "This option determine the OpenCL C language version to use. Support all OpenCL C programs that use the OpenCL C language 1.2 specification." )
    set( OPENCL_C_VERSION_2_0 CACHE BOOL
      "This option determine the OpenCL C language version to use. Support all OpenCL C programs that use the OpenCL C language 2.0 specification." )

    mark_as_advanced( OPENCL_C_VERSION_1_1 )
    mark_as_advanced( OPENCL_C_VERSION_1_2 )
    mark_as_advanced( OPENCL_C_VERSION_2_0 )
  else()
    unset( OPENCL_PROFILING CACHE )

    unset( OPENCL_MATH_SINGLE_PRECISION_CONSTANT CACHE )
    unset( OPENCL_MATH_DENORMS_ARE_ZERO CACHE )
    unset( OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT CACHE )

    unset( OPENCL_OPTIMIZATION_OPT_DISABLE CACHE )
    unset( OPENCL_OPTIMIZATION_MAD_ENABLE CACHE )
    unset( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS CACHE )
    unset( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE )
    unset( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE )
    unset( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH CACHE )
    unset( OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE CACHE )

    unset( OPENCL_WARNINGS_DISABLE CACHE )
    unset( OPENCL_WARNINGS_AS_ERRORS CACHE )

    unset( OPENCL_C_VERSION_1_1 CACHE )
    unset( OPENCL_C_VERSION_1_2 CACHE )
    unset( OPENCL_C_VERSION_2_0 CACHE )
  endif()
endmacro()

# macro opencl_append_options_to_cxx_flags
macro( opencl_append_options_to_cxx_flags )
  # Add extra options for AMD
  if( OPENCL_USE_AMD_SDK )
    # Set OPENCL_MATH_SINGLE_PRECISION_CONSTANT obligingly for AMD
    set( OPENCL_MATH_SINGLE_PRECISION_CONSTANT ON CACHE BOOL
      "Treat double precision floating-point constant as single precision constant." FORCE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_MATH_SINGLE_PRECISION_CONSTANT" )
  else()
    if( OPENCL_MATH_SINGLE_PRECISION_CONSTANT )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_MATH_SINGLE_PRECISION_CONSTANT" )
    endif()
  endif()

  if( OPENCL_MATH_DENORMS_ARE_ZERO )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_MATH_DENORMS_ARE_ZERO" )
  endif()

  if( OPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_MATH_FP32_CORRECTLY_ROUNDED_DIVIDE_SQRT" )
  endif()

  # Add OpenCL Optimization
  if( OPENCL_OPTIMIZATION_OPT_DISABLE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_OPT_DISABLE" )
    unset( OPENCL_OPTIMIZATION_MAD_ENABLE CACHE )
    unset( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS CACHE )
    unset( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE )
    unset( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE )
    unset( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH CACHE )
    unset( OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE CACHE )
  endif()

  if( NOT OPENCL_OPTIMIZATION_OPT_DISABLE )
    if( OPENCL_OPTIMIZATION_MAD_ENABLE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_MAD_ENABLE" )
    endif()

    if( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_NO_SIGNED_ZEROS" )
    endif()

    if( OPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_UNIFORM_WORK_GROUP_SIZE" )
    endif()

    if( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_FAST_RELAXED_MATH" )
      unset( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE )
      unset( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE )
    endif()

    if( NOT OPENCL_OPTIMIZATION_FAST_RELAXED_MATH )
      if( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS" )
      endif()

      if( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_FINITE_MATH_ONLY" )
      endif()
    endif()
  endif()

  # Add OpenCL Warnings
  if( OPENCL_WARNINGS_DISABLE )
    unset( OPENCL_WARNINGS_AS_ERRORS CACHE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_WARNINGS_DISABLE" )
  endif()
  if( OPENCL_WARNINGS_AS_ERRORS )
    unset( OPENCL_WARNINGS_DISABLE CACHE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_WARNINGS_AS_ERRORS" )
  endif()

  # Add Options Controlling the OpenCL C Version
  if( OPENCL_C_VERSION_1_1 )
    unset( OPENCL_C_VERSION_1_2 CACHE )
    unset( OPENCL_C_VERSION_2_0 CACHE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_C_VERSION_1_1" )
  endif()
  if( OPENCL_C_VERSION_1_2 )
    unset( OPENCL_C_VERSION_1_1 CACHE )
    unset( OPENCL_C_VERSION_2_0 CACHE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_C_VERSION_1_2" )
  endif()
  if( OPENCL_C_VERSION_2_0 )
    unset( OPENCL_C_VERSION_1_0 CACHE )
    unset( OPENCL_C_VERSION_1_1 CACHE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_C_VERSION_2_0" )
  endif()

  # Add OpenCL Profiling
  if( OPENCL_PROFILING )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_PROFILING" )
  endif()
endmacro()

#=============================================================================
# FindOpenCL.cmake starts here
opencl_init()                        # Set FindOpenCL.cmake to initial state
opencl_find_intel()                  # Find Intel OpenCL SDK
opencl_find_nvidia()                 # Find NVIDIA OpenCL SDK
opencl_find_amd()                    # Find AMD OpenCL SDK
opencl_define_avaliable_sdk()        # Define OpenCL SDK's
opencl_select_sdk()                  # Select OpenCL SDK
opencl_select_default_sdk()          # Perform selecting default OpenCL SDK
opencl_define_options()              # Define OpenCL options
opencl_append_options_to_cxx_flags() # Append OpenCL options to CMAKE_CXX_FLAGS

# handle the QUIETLY and REQUIRED arguments and set OpenCL_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )

mark_as_advanced( OPENCL_INCLUDE_DIRS )
mark_as_advanced( OPENCL_LIBRARIES )
