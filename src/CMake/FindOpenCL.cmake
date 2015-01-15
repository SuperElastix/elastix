# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD / ATI, Apple and NVIDIA implementations, but should work, too.
#
# Once done this will define
#  OPENCL_FOUND         - system has OpenCL
#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory
#  OPENCL_LIBRARIES     - link these to use OpenCL

find_package( PackageHandleStandardArgs )

# Platforms
set( OPENCL_USE_PLATFORM_INTEL FALSE CACHE BOOL  "Use Intel implementation of the OpenCL.")
set( OPENCL_USE_PLATFORM_NVIDIA FALSE CACHE BOOL "Use NVidia implementation of the OpenCL.")
set( OPENCL_USE_PLATFORM_AMD FALSE CACHE BOOL    "Use AMD implementation of the OpenCL.")

# Define profiling variable for ITK4OpenCL
set( OPENCL_PROFILING OFF CACHE BOOL
  "Enable OpenCL profiling with CL_QUEUE_PROFILING_ENABLE. Event objects can be used to capture profiling information that measure execution time of a command." )

# OpenCL Math Intrinsics Options
if( OPENCL_USE_PLATFORM_INTEL OR OPENCL_USE_PLATFORM_NVIDIA OR OPENCL_USE_PLATFORM_AMD )
  set( OPENCL_MATH_SINGLE_PRECISION_CONSTANT CACHE BOOL
    "Treat double precision floating-point constant as single precision constant." )
  set( OPENCL_MATH_DENORMS_ARE_ZERO CACHE BOOL
    "This option controls how single precision and double precision denormalized numbers are handled." )

  mark_as_advanced( OPENCL_MATH_SINGLE_PRECISION_CONSTANT )
  mark_as_advanced( OPENCL_MATH_DENORMS_ARE_ZERO )
endif()

# OpenCL Optimization Options
if( OPENCL_USE_PLATFORM_INTEL OR OPENCL_USE_PLATFORM_NVIDIA OR OPENCL_USE_PLATFORM_AMD )
  set( OPENCL_OPTIMIZATION_OPT_DISABLE CACHE BOOL
    "This option disables all optimizations. The default is optimizations are enabled." )
  set( OPENCL_OPTIMIZATION_STRICT_ALIASING CACHE BOOL
    "This option allows the compiler to assume the strictest aliasing rules." )
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

  mark_as_advanced( OPENCL_OPTIMIZATION_OPT_DISABLE )
  mark_as_advanced( OPENCL_OPTIMIZATION_STRICT_ALIASING )
  mark_as_advanced( OPENCL_OPTIMIZATION_MAD_ENABLE )
  mark_as_advanced( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS )
  mark_as_advanced( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS )
  mark_as_advanced( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY )
  mark_as_advanced( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH )
endif()

# OpenCL Options to Request or Suppress Warnings
if( OPENCL_USE_PLATFORM_INTEL OR OPENCL_USE_PLATFORM_NVIDIA OR OPENCL_USE_PLATFORM_AMD )
  set( OPENCL_WARNINGS_DISABLE CACHE BOOL
    "This option inhibit all warning messages." )
  set( OPENCL_WARNINGS_AS_ERRORS CACHE BOOL
    "This option make all warnings into errors." )
endif()

# OpenCL Options Controlling the OpenCL C Version
if( OPENCL_USE_PLATFORM_INTEL OR OPENCL_USE_PLATFORM_NVIDIA OR OPENCL_USE_PLATFORM_AMD )
  set( OPENCL_C_VERSION_1_1 CACHE BOOL
    "This option determine the OpenCL C language version to use. Support all OpenCL C programs that use the OpenCL C language 1.1 specification." )
  set( OPENCL_C_VERSION_1_2 CACHE BOOL
    "This option determine the OpenCL C language version to use. Support all OpenCL C programs that use the OpenCL C language 1.2 specification." )
endif()

# Apple here
if( APPLE )
endif()

# If cl.h is not in the OPENCL_INCLUDE_DIRS then set
# OPENCL_INCLUDE_DIRS to NOTFOUND
if( NOT EXISTS ${OPENCL_INCLUDE_DIRS}/CL/cl.h )
  set( OPENCL_INCLUDE_DIRS OPENCL_INCLUDE_DIRS-NOTFOUND CACHE PATH
    "OpenCL path to CL/cl.h include directory" FORCE )
endif()

if( UNIX )
  # If OpenCL is not in the OPENCL_LIBRARIES then set
  # OPENCL_LIBRARIES to NOTFOUND
  #set( OPENCL_LIBRARIES "")
  #message(STATUS "OPENCL_LIBRARIES: " ${OPENCL_LIBRARIES})
  string( FIND ${OPENCL_LIBRARIES} "OpenCL" OPENCL_LIBRARIES_EXIST )
  if( OPENCL_LIBRARIES_EXIST EQUAL -1 )
    set( OPENCL_LIBRARIES OPENCL_LIBRARIES-NOTFOUND CACHE PATH
      "OpenCL path to OpenCL library" FORCE )
  endif()
endif()

if( WIN32 )
  # If OpenCL.lib is not in the OPENCL_LIBRARIES then set
  # OPENCL_LIBRARIES to NOTFOUND
  string( FIND ${OPENCL_LIBRARIES} "OpenCL.lib" OPENCL_LIBRARIES_EXIST )
  if( OPENCL_LIBRARIES_EXIST EQUAL -1 )
    set( OPENCL_LIBRARIES OPENCL_LIBRARIES-NOTFOUND CACHE PATH
      "OpenCL path to OpenCL.lib library" FORCE )
  endif()
endif()

# Apple, NOT TESTED
if( APPLE )
  find_library( OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX" )
  find_path( OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX" )
  find_path( _OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX" )

  # Unix style platforms
  find_library( OPENCL_LIBRARIES OpenCL ENV LD_LIBRARY_PATH )

  get_filename_component( OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH )
  get_filename_component( _OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE )

  # The AMD SDK currently does not place its headers
  # in /usr/include, therefore also search relative
  # to the library
  find_path( OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} /usr/local/cuda/include/ )
  find_path( _OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} )
endif()

# Windows
if( WIN32 OR UNIX )
  # If none of the platform is set
  if( NOT OPENCL_USE_PLATFORM_INTEL AND NOT OPENCL_USE_PLATFORM_NVIDIA AND NOT OPENCL_USE_PLATFORM_AMD )
    set( OPENCL_INCLUDE_DIRS OPENCL_INCLUDE_DIRS-NOTFOUND CACHE PATH
      "OpenCL path to CL/cl.h include directory" FORCE )
    set( OPENCL_LIBRARIES OPENCL_LIBRARIES-NOTFOUND CACHE PATH
      "OpenCL path to OpenCL.lib library" FORCE )

    # Unset an extra AMD/Intel option
    unset( OPENCL_USE_PLATFORM_AMD_GPU_CPU CACHE )
    unset( OPENCL_USE_PLATFORM_INTEL_GPU_CPU CACHE )

    # Unset optimizations
    unset( OPENCL_OPTIMIZATION_OPT_DISABLE CACHE )
    unset( OPENCL_OPTIMIZATION_STRICT_ALIASING CACHE )
    unset( OPENCL_OPTIMIZATION_MAD_ENABLE CACHE )
    unset( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS CACHE )
    unset( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE )
    unset( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE )
    unset( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH CACHE )

    # Unset warnings
    unset( OPENCL_WARNINGS_DISABLE CACHE )
    unset( OPENCL_WARNINGS_AS_ERRORS CACHE )

    # Unset options controlling the OpenCL C version
    unset( OPENCL_C_VERSION_1_1 CACHE )
    unset( OPENCL_C_VERSION_1_2 CACHE )

    set( OPENCL_PLATFORM_STRING "OpenCL not found" CACHE INTERNAL "OpenCL Platform" )

    message( WARNING "Please select OpenCL platform" )
  endif()

  # Intel OpenCL
  if( OPENCL_USE_PLATFORM_INTEL )
    set( OPENCL_USE_PLATFORM_INTEL_GPU_CPU TRUE CACHE BOOL
      "Use Intel GPU or CPU implementation of the OpenCL. On use GPU, Off use CPU" )

    # Unset other platforms
    unset( OPENCL_USE_PLATFORM_NVIDIA CACHE )
    unset( OPENCL_USE_PLATFORM_AMD CACHE )

    # Find Intel OpenCL SDK
    if( ${OPENCL_INCLUDE_DIRS} STREQUAL "OPENCL_INCLUDE_DIRS-NOTFOUND" )
      find_path( OPENCL_INCLUDE_DIRS CL/cl.h
        HINTS $ENV{INTELOCLSDKROOT}
        PATH_SUFFIXES include
        PATHS /usr/local/intel )
    endif()

    if( ${OPENCL_LIBRARIES} STREQUAL "OPENCL_LIBRARIES-NOTFOUND" )
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        find_library( OPENCL_LIBRARIES lib/x64/OpenCL.lib $ENV{INTELOCLSDKROOT} )
      else()
        find_library( OPENCL_LIBRARIES lib/x86/OpenCL.lib $ENV{INTELOCLSDKROOT} )
      endif()
    endif()

    if( ${OPENCL_USE_PLATFORM_INTEL_GPU_CPU} )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_INTEL_GPU_OPENCL" )
    else()
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_INTEL_CPU_OPENCL" )
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_INTEL_OPENCL" )
    set( OPENCL_PLATFORM_STRING "Intel OpenCL" CACHE INTERNAL "OpenCL Platform" )

  endif() # Intel OpenCL

  # NVidia OpenCL
  if( OPENCL_USE_PLATFORM_NVIDIA )

    # Unset other platforms
    unset( OPENCL_USE_PLATFORM_INTEL CACHE )
    unset( OPENCL_USE_PLATFORM_AMD CACHE )

    # Find NVidia OpenCL SDK
    if( ${OPENCL_INCLUDE_DIRS} STREQUAL "OPENCL_INCLUDE_DIRS-NOTFOUND" )
      find_path( OPENCL_INCLUDE_DIRS CL/cl.h
        HINTS
          $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/inc
          $ENV{CUDA_PATH_V5_0}
          $ENV{CUDA_PATH_V4_0}
          $ENV{CUDA_PATH_V3_2}
        PATH_SUFFIXES include
        PATHS /usr/local/cuda )
    endif()

    if( ${OPENCL_LIBRARIES} STREQUAL "OPENCL_LIBRARIES-NOTFOUND" )
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        find_library( OPENCL_LIBRARIES
          NAMES OpenCL
          HINTS
            $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common
            $ENV{CUDA_PATH_V5_0}
            $ENV{CUDA_PATH_V4_0}
            $ENV{CUDA_PATH_V3_2}
          PATH_SUFFIXES lib/x64
          PATHS /usr/local/cuda )
      else()
        find_library( OPENCL_LIBRARIES
          NAMES OpenCL
          HINTS
            $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common
            $ENV{CUDA_PATH_V5_0}
            $ENV{CUDA_PATH_V4_0}
            $ENV{CUDA_PATH_V3_2}
          PATH_SUFFIXES lib/Win32
          PATHS /usr/local/cuda )
      endif()
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_NVIDIA_OPENCL" )
    set( OPENCL_PLATFORM_STRING "NVidia OpenCL" CACHE INTERNAL "OpenCL Platform" )

  endif() # NVidia OpenCL

  # AMD OpenCL
  if( OPENCL_USE_PLATFORM_AMD )
    set( OPENCL_USE_PLATFORM_AMD_GPU_CPU TRUE CACHE BOOL
      "Use AMD GPU or CPU implementation of the OpenCL. On use GPU, Off use CPU" )

    # Unset other platforms
    unset( OPENCL_USE_PLATFORM_INTEL CACHE )
    unset( OPENCL_USE_PLATFORM_NVIDIA CACHE )

    # Find AMD OpenCL SDK
    if( ${OPENCL_INCLUDE_DIRS} STREQUAL "OPENCL_INCLUDE_DIRS-NOTFOUND" )
      find_path( OPENCL_INCLUDE_DIRS CL/cl.h
        HINTS $ENV{AMDAPPSDKROOT}
        PATH_SUFFIXES include
        PATHS /usr/local )
    endif()

    if( ${OPENCL_LIBRARIES} STREQUAL "OPENCL_LIBRARIES-NOTFOUND" )
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
        find_library( OPENCL_LIBRARIES
          NAMES OpenCL
          HINTS $ENV{AMDAPPSDKROOT}
          PATH_SUFFIXES lib/x86_64
          PATHS /usr/local )
      else()
        find_library( OPENCL_LIBRARIES
          NAMES OpenCL
          HINTS $ENV{AMDAPPSDKROOT}
          PATH_SUFFIXES lib/x86
          PATHS /usr/local )
      endif()
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_AMD_OPENCL -DATI_OS_WIN" )

    if( OPENCL_USE_PLATFORM_AMD_GPU_CPU )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_AMD_GPU_OPENCL" )
    else()
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DITK_USE_AMD_CPU_OPENCL" )
    endif()

    set( OPENCL_PLATFORM_STRING "AMD OpenCL" CACHE INTERNAL "OpenCL Platform" )

  endif() # AMD OpenCL

endif()

# Add OpenCL Math Intrinsics Options
if( OPENCL_USE_PLATFORM_AMD )
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

# Add OpenCL Optimization
if( OPENCL_OPTIMIZATION_OPT_DISABLE )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_OPT_DISABLE" )
  unset( OPENCL_OPTIMIZATION_STRICT_ALIASING CACHE )
  unset( OPENCL_OPTIMIZATION_MAD_ENABLE CACHE )
  unset( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS CACHE )
  unset( OPENCL_OPTIMIZATION_UNSAFE_MATH_OPTIMIZATIONS CACHE )
  unset( OPENCL_OPTIMIZATION_FINITE_MATH_ONLY CACHE )
  unset( OPENCL_OPTIMIZATION_FAST_RELAXED_MATH CACHE )
endif()

if( NOT OPENCL_OPTIMIZATION_OPT_DISABLE )
  if( OPENCL_OPTIMIZATION_STRICT_ALIASING )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_STRICT_ALIASING" )
  endif()

  if( OPENCL_OPTIMIZATION_MAD_ENABLE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_MAD_ENABLE" )
  endif()

  if( OPENCL_OPTIMIZATION_NO_SIGNED_ZEROS )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_OPTIMIZATION_NO_SIGNED_ZEROS" )
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
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_C_VERSION_1_1" )
endif()
if( OPENCL_C_VERSION_1_2 )
  unset( OPENCL_C_VERSION_1_1 CACHE )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_C_VERSION_1_2" )
endif()

# Add OpenCL Profiling
if( OPENCL_PROFILING )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCL_PROFILING" )
endif()

# handle the QUIETLY and REQUIRED arguments and set OpenCL_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )

mark_as_advanced( OPENCL_INCLUDE_DIRS )
mark_as_advanced( OPENCL_LIBRARIES )

