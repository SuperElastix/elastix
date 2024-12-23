# Minimum CMake version. Equal to ITK 5.3 ITK_OLDEST_VALIDATED_POLICIES_VERSION.
cmake_minimum_required(VERSION 3.16.3)

project(elastix)
set(CMAKE_CXX_STANDARD 17)

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Elastix does not support BUILD_SHARED_LIBS")
endif()

#---------------------------------------------------------------------
cmake_policy(SET CMP0012 NEW) # "if() recognizes numbers and boolean constants"
cmake_policy(SET CMP0042 NEW) # "MACOSX_RPATH is enabled by default."

#---------------------------------------------------------------------
include(CTest)

#---------------------------------------------------------------------
# Allow specifying whether or not the executables are built. 
option( ELASTIX_BUILD_EXECUTABLE "Build elastix and transformix as executable? (The libraries are always built as well anyway.)" ON )

  # The following may make smaller and quicker loading libraries,
  # that hides unnecessary symbols. Available from CMake 3.0.0.
  #set(CMAKE_C_VISIBILITY_PRESET hidden)
  #set(CMAKE_CXX_VISIBILITY_PRESET hidden)

#---------------------------------------------------------------------
option(ELASTIX_USE_OPENCL "Use OpenCL enabled GPU" OFF)
set(_GPU_depends "")
if(ELASTIX_USE_OPENCL)
  list(APPEND _GPU_depends ITKGPUCommon)
endif()

# Find ITK.
find_package(ITK 5.4.1 REQUIRED COMPONENTS
    ITKCommon
    ITKDisplacementField
    ITKDistanceMap
    ITKGDCM
    ITKImageCompose
    ITKImageFunction
    ITKImageGradient
    ITKImageGrid
    ITKImageIntensity
    ITKImageStatistics
    ITKIOImageBase
    ITKIOMeshBase
    ITKIOMeshOBJ
    ITKIOMeta
    ITKIOTransformBase
    ITKIOXML
    ITKMathematicalMorphology
    ITKMesh
    ITKOptimizers
    ITKRegistrationCommon
    ITKSmoothing
    ITKSpatialObjects
    ITKStatistics
    ITKTestKernel
    ITKThresholding
    ITKTransform
    ITKTransformFactory
    ${_GPU_depends}
    ITKImageIO
    ITKTransformIO
    ITKMeshIO
    )
include(${ITK_USE_FILE})

#---------------------------------------------------------------------
# Add the CMake dir as an additional module path,
# so that CMake is able to find the FindPackage modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

#---------------------------------------------------------------------
# Get version information.
include(elastixVersion)

# Function for exporting targets
include(elastixExportTarget)

if (MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(
    -Woverloaded-virtual
    -Wshadow
    -Wunused-parameter
    )
endif()

#---------------------------------------------------------------------
# Find OpenCL.
if(ELASTIX_USE_OPENCL)
  find_package(OpenCL REQUIRED QUIET)

  # Make sure ITK is not compiled with GPU extensions on,
  # since elastix overrides ITK classes with new ones.
  if(${ITK_USE_GPU})
    message(FATAL_ERROR "ITK_USE_GPU: " ${ITK_USE_GPU}
      "\nERROR: ITK should be compiled with ITK_USE_GPU OFF, as elastix overrides ITK classes.")
  endif()

  # Force OPENCL_OPTIMIZATION_MAD_ENABLE to on
  if(DEFINED OPENCL_OPTIMIZATION_MAD_ENABLE)
    set(OPENCL_OPTIMIZATION_MAD_ENABLE ON CACHE BOOL
      "Allow a * b + c to be replaced by a mad. The mad computes a * b + c with reduced accuracy." FORCE)
  endif()

  # Include the OpenCL include directories to elastix
  include_directories(${OPENCL_INCLUDE_DIRS})

  # Add some useful macro's
  include(elastixOpenCL)

  # Add definition for the OpenCL
  add_definitions(-DELASTIX_USE_OPENCL)
endif()

#---------------------------------------------------------------------
# Find Eigen
mark_as_advanced(ELASTIX_USE_EIGEN)
option(ELASTIX_USE_EIGEN "Use Eigen library." OFF)

if(ELASTIX_USE_EIGEN)
  find_package(Eigen3 REQUIRED)
  include_directories(${EIGEN3_INCLUDE_DIR})
  add_definitions(-DELASTIX_USE_EIGEN)
endif()

#---------------------------------------------------------------------
# Set single (build-tree) output directories for all executables and libraries.
# This makes it easier to create an elxUseFile.cmake, that users can
# include in their programs to borrow elastix functionality.

if(NOT LIBRARY_OUTPUT_PATH)
  set(LIBRARY_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL
    "Single output directory for building all libraries.")
endif()

if(NOT EXECUTABLE_OUTPUT_PATH)
  set(EXECUTABLE_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL
    "Single output directory for building all executables.")
endif()

# Mark these variables as advanced; their default value is usually fine
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

#---------------------------------------------------------------------
# Define the install directory names, relative to install prefix
# given by CMAKE_INSTALL_PREFIX.  All components should use the
# same name.  Set as variable here for future customization; e.g.
# if desire to decorate with Elastix version.
# Allow user to customize.

mark_as_advanced(ELASTIX_ARCHIVE_DIR ELASTIX_INCLUDE_DIR ELASTIX_LIBRARY_DIR ELASTIX_RUNTIME_DIR)
set(ELASTIX_ARCHIVE_DIR "lib"      CACHE STRING
  "Directory for installing archive files; path is relative to CMAKE_INSTALL_PREFIX")
set(ELASTIX_INCLUDE_DIR "include"  CACHE STRING
  "Directory for installing include files; path is relative to CMAKE_INSTALL_PREFIX")
set(ELASTIX_LIBRARY_DIR "lib"      CACHE STRING
  "Directory for installing library files; path is relative to CMAKE_INSTALL_PREFIX")
set(ELASTIX_RUNTIME_DIR "bin"      CACHE STRING
  "Directory for installing runtime files; path is relative to CMAKE_INSTALL_PREFIX")
if(NOT ELASTIX_INSTALL_PACKAGE_DIR)
  set(ELASTIX_INSTALL_PACKAGE_DIR "${ELASTIX_LIBRARY_DIR}/cmake/elastix")
endif()

#---------------------------------------------------------------------
# Provide options to avoid installing runtime or development components
option(ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES "Do not install runtime libraries" OFF)
option(ELASTIX_NO_INSTALL_EXECUTABLES "Do not install executables" OFF)
option(ELASTIX_NO_INSTALL_DEVELOPMENT "Do not install development headers and static libraries" OFF)
mark_as_advanced(ELASTIX_NO_INSTALL_EXECUTABLES ELASTIX_NO_INSTALL_RUNTIME_LIBRARIES ELASTIX_NO_INSTALL_DEVELOPMENT)

#---------------------------------------------------------------------
# Check if Mevis DicomTiff support is desired
mark_as_advanced(ELASTIX_USE_MEVISDICOMTIFF)
option(ELASTIX_USE_MEVISDICOMTIFF
  "Support MevisLab DicomTiff image format" OFF)

#---------------------------------------------------------------------
# Define cmake variable to define extra user component directories
# These directories will be added to the list of include directories
# and they will be searched for CMakeLists.txt files for further
# processing. In these directories, users may put code for their own
# components.

mark_as_advanced(ELASTIX_USER_COMPONENT_DIRS)
set(ELASTIX_USER_COMPONENT_DIRS "" CACHE PATH
  "directories with user defined elastix components")

#---------------------------------------------------------------------
# If IDE supports it, do use folder view.
# gcc automatically ignores it. VC Express does not support and gives
# annoying warnings when this option is turned on.
# VC pro does support it.
set(CMAKE_USE_FOLDERS ON CACHE INTERNAL "Use folder view in IDE")
if(CMAKE_MAKE_PROGRAM MATCHES ".?VCExpress.?")
  set(CMAKE_USE_FOLDERS OFF CACHE INTERNAL "Use folder view in IDE")
endif()
set_property(GLOBAL PROPERTY USE_FOLDERS ${CMAKE_USE_FOLDERS})

#---------------------------------------------------------------------
# Set default build type to Release, if none was specified
# Taken from ITK CMake list
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#---------------------------------------------------------------------
# Include directories

set(elxCommon_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Common
  ${elastix_SOURCE_DIR}/Common/CostFunctions
  ${elastix_SOURCE_DIR}/Common/ImageSamplers
  ${elastix_SOURCE_DIR}/Common/LineSearchOptimizers
  ${elastix_SOURCE_DIR}/Common/ParameterFileParser
  ${elastix_SOURCE_DIR}/Common/Transforms
  ${elastix_SOURCE_DIR}/Common/MevisDicomTiff
)

set(elxCommon_OpenCL_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Common/OpenCL
  ${elastix_SOURCE_DIR}/Common/OpenCL/Factories
  ${elastix_SOURCE_DIR}/Common/OpenCL/ITKimprovements
  ${elastix_SOURCE_DIR}/Common/OpenCL/Filters
  ${elastix_SOURCE_DIR}/Common/OpenCL/Kernels
)

set(elxCore_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Core
  ${elastix_SOURCE_DIR}/Core/Install
  ${elastix_SOURCE_DIR}/Core/Kernel
  ${elastix_SOURCE_DIR}/Core/ComponentBaseClasses
  ${elastix_SOURCE_DIR}/Core/Configuration
  ${elastix_SOURCE_DIR}/Core/Main
)

set(elxComponents_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Components/FixedImagePyramids
  ${elastix_SOURCE_DIR}/Components/ImageSamplers
  ${elastix_SOURCE_DIR}/Components/Interpolators
  ${elastix_SOURCE_DIR}/Components/Metrics
  ${elastix_SOURCE_DIR}/Components/MovingImagePyramids
  ${elastix_SOURCE_DIR}/Components/Optimizers
  ${elastix_SOURCE_DIR}/Components/Registrations
  ${elastix_SOURCE_DIR}/Components/ResampleInterpolators
  ${elastix_SOURCE_DIR}/Components/Resamplers
  ${elastix_SOURCE_DIR}/Components/Transforms
)

set(elxINCLUDE_DIRECTORIES
  ${elxCommon_INCLUDE_DIRECTORIES}
  ${elxCore_INCLUDE_DIRECTORIES}
  ${elxComponents_INCLUDE_DIRECTORIES}
  ${elastix_BINARY_DIR}
  ${ELASTIX_USER_COMPONENT_DIRS}
)

include_directories(${elxINCLUDE_DIRECTORIES})

# include the OpenCL directories
# The key-word BEFORE is important here, because the elastix distribution
# contains some files with the same name as in the ITK distribution.
# Some bugs were fixed and features added. When these are contributed back
# to the ITK the BEFORE keyword is not needed anymore.
if(ELASTIX_USE_OPENCL)
  include_directories(BEFORE ${elxCommon_OpenCL_INCLUDE_DIRECTORIES})
  list(APPEND elxINCLUDE_DIRECTORIES ${elxCommon_OpenCL_INCLUDE_DIRECTORIES})
endif()

#---------------------------------------------------------------------
# Microsoft specific items

if(MSVC)
  # Kill the anoying MS VS warning about non-safe functions.
  # They hide real warnings.
  add_definitions(
    /D_SCL_SECURE_NO_DEPRECATE
    /D_CRT_SECURE_NO_DEPRECATE
    /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE)

  # Increases address capacity
  if(WIN32)
    add_compile_options(/bigobj)
  endif()
endif()

#---------------------------------------------------------------------
# Process the sub-directories

# Common dir: code that is neither related to the core of elastix or
# to specific components.
add_subdirectory(Common)

# Components: the registration components such as metrics, transforms,
# optimizers, etc.
add_subdirectory(Components)

# Core dir: code that takes care of starting elastix, loading
# components, definitions of macros etc.
add_subdirectory(Core)

#---------------------------------------------------------------------
# Configure the examples

set(ELASTIX_DOX_DIR  ${elastix_SOURCE_DIR}/dox)
set(ELASTIX_TOOLS_DIR  ${elastix_BINARY_DIR}/tools)
set(ELASTIX_HELP_DIR ${elastix_BINARY_DIR}/help CACHE PATH
  "path to the doxygen generated help files and the examples")

# Copy the examples to the help directory

if(WIN32)
  configure_file(
    ${ELASTIX_DOX_DIR}/example.bat
    ${ELASTIX_HELP_DIR}/example.bat
    COPYONLY)
else()
  configure_file(
    ${ELASTIX_DOX_DIR}/example
    ${ELASTIX_HELP_DIR}/example
    COPYONLY)
endif()

make_directory(${ELASTIX_HELP_DIR}/exampleinput)
make_directory(${ELASTIX_HELP_DIR}/exampleoutput)

set(ELX_EXAMPLEINPUTFILES
  fixed.mhd
  fixed.raw
  mask_fixed.mhd
  mask_fixed.raw
  mask_moving.mhd
  mask_moving.raw
  moving.mhd
  moving.raw
  parameters_Affine.txt
  parameters_BSpline.txt
  parameters_Rigid.txt
  parameters_Translation.txt
  solution_deformedmovingimage.mhd
  solution_deformedmovingimage.raw
)

foreach(ELX_EXAMPLEINPUTFILE ${ELX_EXAMPLEINPUTFILES})
  configure_file(
    ${ELASTIX_DOX_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    ${ELASTIX_HELP_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    COPYONLY)
endforeach()

#---------------------------------------------------------------------
# Configure the doxygen-configuration

option(BUILD_DOCUMENTATION "Build elastix documentation." OFF)

if(${BUILD_DOCUMENTATION})
  find_package(Doxygen QUIET)
  string(COMPARE NOTEQUAL ${DOXYGEN} "DOXYGEN-NOTFOUND" Doxygen_FOUND)
  if(Doxygen_FOUND)
    # Set the path to the doxygen configuration source
    set(ELASTIX_DOXYGEN_DIR ${ELASTIX_DOX_DIR}/doxygen)

    # Get the version number of doxygen
    exec_program(${DOXYGEN} ARGS "--version" OUTPUT_VARIABLE ELASTIX_DOXYGEN_VERSION)

    # Get date
    if(UNIX OR CYGWIN)
      exec_program("date '+%d-%m-%Y'" OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATE)
    endif()
    if(WIN32)
      if(NOT CYGWIN AND NOT MINGW)
        execute_process(COMMAND "${ELASTIX_DOXYGEN_DIR}/doxdate.bat"
          OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATETEMP)
        string(REPLACE "/" "-" ELASTIX_DOXYGEN_DATE ${ELASTIX_DOXYGEN_DATETEMP})
      endif()
    endif()

    # Configure the doxygen configuration
    configure_file(
      ${ELASTIX_DOXYGEN_DIR}/doxyfile.in
      ${ELASTIX_HELP_DIR}/doxyfile.out @ONLY)

    # Configure the footer of the help website.
    configure_file(
      ${ELASTIX_DOXYGEN_DIR}/DoxygenFooter.html.in
      ${ELASTIX_HELP_DIR}/DoxygenFooter.html @ONLY)

    # Configure the MainPage.dox
    configure_file(
      ${ELASTIX_DOXYGEN_DIR}/MainPage.dox.in
      ${ELASTIX_HELP_DIR}/MainPage.dox @ONLY)
  endif()
endif()

#---------------------------------------------------------------------
# Testing

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

mark_as_advanced(ELASTIX_USE_GTEST)
option(ELASTIX_USE_GTEST "Use GoogleTest to test Elastix implementation" OFF)

if(ELASTIX_USE_GTEST)
  enable_testing()
  add_subdirectory(Common/GTesting)
  add_subdirectory(Core/Main/GTesting)
endif()

#---------------------------------------------------------------------
# Packaging

mark_as_advanced(ELASTIX_ENABLE_PACKAGER)
option(ELASTIX_ENABLE_PACKAGER "Enable elastix packager using CPack" OFF)

if(ELASTIX_ENABLE_PACKAGER)
  # Version information
  # If the next line is uncommented the package name will be like
  # elastix-4.3-win64 instead of elastix-4.3.0-win64
  set(CPACK_PACKAGE_VERSION_MAJOR ${ELASTIX_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${ELASTIX_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${ELASTIX_VERSION_PATCH})

  # Also install the copyright file, since when the user enables packaging
  # we assume that the package is meant to distribute.
  install(FILES "${elastix_SOURCE_DIR}/LICENSE" DESTINATION .)
  install(FILES "${elastix_SOURCE_DIR}/NOTICE" DESTINATION .)

  # We have split the elastix package into two components:
  # - the core: elastix and transformix
  # - the libraries: ANNlib
  # NOTE: Currently does not work for nsis. A bug related to this
  # seems to be fixed in the upcoming CMake 2.8.3
  # Therefore, disable component support for now.
  #set(CPACK_COMPONENTS_ALL core libraries Unspecified)
  #  CPACK_ADD_COMPONENT( core
  #  "Core files"
  #  DESCRIPTION "Contains elastix and transformix"
  #  REQUIRED
  #  DEPENDS libraries)
  #   [GROUP group]
  #   [DEPENDS comp1 comp2 ... ]
  #   [INSTALL_TYPES type1 type2 ... ]
  #   [DOWNLOADED]
  #   [ARCHIVE_FILE filename])
  #CPACK_ADD_COMPONENT( libraries
  #  "Libraries"
  #  DESCRIPTION "Contains the libraries"
  #  REQUIRED)
#set(CPACK_COMPONENT_CORE_DISPLAY_NAME "Core files")
#set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
#set(CPACK_COMPONENT_CORE_DESCRIPTION "Contains elastix and transformix")
#set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Contains the libraries")
#set(CPACK_COMPONENT_CORE_DEPENDS libraries)
#set(CPACK_COMPONENT_CORE_REQUIRED ON)
#set(CPACK_COMPONENT_LIBRARIES_REQUIRED ON)

  # The default package filename is
  # ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}
  # which is like elastix-4.3.0-win64, or elastix-4.3.0-linux-i686
  # Currently however we use elastix_windows64_v4.3
  # We can change our naming schedule or come close to it using:
  #set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_SYSTEM_NAME}_v${CPACK_PACKAGE_VERSION}")
  # but it doesn't work since these variables are not defined yet
  # Moving include(CPack) to above introduces other errors.
  #set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}_${CMAKE_SYSTEM_NAME}_v${ELASTIX_VERSION}")
  # also does not work properly. Just use the default for now.

  # General information
  set(CPACK_PACKAGE_VENDOR "Stefan Klein and Marius Staring")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "elastix is an image registration toolkit")
  #set(CPACK_PACKAGE_DESCRIPTION_FILE
  #  "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
  set(CPACK_RESOURCE_FILE_LICENSE
    "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE")

  # The default install directories: .../elastix_v4.3
  set(CPACK_PACKAGE_INSTALL_DIRECTORY
    "elastix_v${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR}")
  set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
    "elastix_v${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR}")

  # Do not include a subdirectory in the zip
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0)

  set(CPACK_SET_DESTDIR "OFF")

  # Make sure the directory structure is kept in the source zip
  # and that also the dox and tools directories are included.
  set(CPACK_SOURCE_INSTALLED_DIRECTORIES
    "${elastix_SOURCE_DIR};/src;${ELASTIX_DOX_DIR};/dox;${ELASTIX_TOOLS_DIR};/tools")

  # ??
  #set(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")

  # For the windows nsis installer only (is this if-check necessary?)
  if(WIN32 AND NOT UNIX)
    # NOTE: There is a bug in NSI that does not handle full unix paths properly.
    # Make sure there is at least one set of four (4) backlashes
    # (CMake escapes 2, and the other gets escaped too in some second step)

    # Set the generators. If left blank the user has all options.
    set(CPACK_GENERATOR "NSIS;ZIP")

    # Adding information
    set(CPACK_NSIS_DISPLAY_NAME
      "${CPACK_PACKAGE_INSTALL_DIRECTORY} elastix")
    set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\elastix.dev")
    set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\elastix.dev")
    set(CPACK_NSIS_CONTACT "elastix@bigr.nl")
    set(CPACK_NSIS_PACKAGE_NAME "elastix")
    set(CPACK_NSIS_DISPLAY_NAME "elastix")

    # Adding icons and images to make it look nice:
    # 1 A branding image that will be displayed inside the installer
    # 2 The icon file(.ico) for the generated install program
    # 3 The icon file(.ico) for the generated uninstall program
    set(CPACK_PACKAGE_ICON
      "${CMAKE_CURRENT_SOURCE_DIR}/dox/art\\\\elastix_logo_full_small.bmp")
    set(CPACK_NSIS_MUI_ICON
      "${CMAKE_CURRENT_SOURCE_DIR}/dox/art\\\\elastix_logo_64.ico")
    set(CPACK_NSIS_MUI_UNIICON
      "${CMAKE_CURRENT_SOURCE_DIR}/dox/art\\\\elastix_logo_64.ico")

    # Create an option in the installer that asks if elastix
    # needs to be added to the system path
    set(CPACK_NSIS_MODIFY_PATH ON)

  else()
    # set the generators
    set(CPACK_GENERATOR "TBZ2;ZIP")

    # set(CPACK_STRIP_FILES "bin/MyExecutable")
    #set(CPACK_SOURCE_STRIP_FILES "")
  endif()

  # Uncomment the following line if we want to include the system
  # dll's in the distribution!
  #include(InstallRequiredSystemLibraries)

  # This include will do all the work.
  include(CPack)
endif()

#---------------------------------------------------------------------
# Make it easier to include elastix functionality in other programs.


# The build settings file. (necessary for elastix?)
#set(ITK_BUILD_SETTINGS_FILE ${ITK_BINARY_DIR}/ITKBuildSettings.cmake)

elastix_export_target(elastix_lib)
elastix_export_target(transformix_lib)

foreach(LIB IN LISTS AllComponentLibs)
  elastix_export_target(${LIB})
endforeach()

# ELASTIX_ITK_DIR -- used (only) for the Config.cmake file in the build tree
# It appears necessary for Windows Azure Pipelines, to replace backslashes by
# forward slashes before using the ITK bin directory.
string(REPLACE "\\" "/" ELASTIX_ITK_DIR ${ITK_DIR})

# Create the Config.cmake file for the build tree:
set(ElastixConfig_TREE "build")
set(ElastixConfig_CODE "if(NOT DEFINED ITK_DIR)
  set(ITK_DIR \"${ELASTIX_ITK_DIR}\")
endif()
")
set(ELX_CONFIG_INCLUDE_DIRECTORIES "${elxINCLUDE_DIRECTORIES}")
configure_file(ElastixConfig.cmake.in ElastixConfig.cmake @ONLY)

# Create the Config.cmake file for the install tree:
set(ElastixConfig_TREE "install")
set(ElastixConfig_CODE "set(Elastix_INSTALL_PREFIX \"\${_ELASTIXConfig_DIR}\")")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${ELASTIX_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(ElastixConfig_CODE "${ElastixConfig_CODE}
get_filename_component(Elastix_INSTALL_PREFIX \"\${Elastix_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(ELX_CONFIG_INCLUDE_DIRECTORIES "\${Elastix_INSTALL_PREFIX}/include")
foreach(ELX_INCLUDE_DIR IN ITEMS ${elxINCLUDE_DIRECTORIES})
  if (NOT ELX_INCLUDE_DIR STREQUAL CMAKE_BINARY_DIR)
    string(REPLACE ${CMAKE_SOURCE_DIR} "" ELX_INCLUDE_DIR ${ELX_INCLUDE_DIR})
    list(APPEND ELX_CONFIG_INCLUDE_DIRECTORIES "\${Elastix_INSTALL_PREFIX}/include${ELX_INCLUDE_DIR}")
  endif()
endforeach()

# The Config.cmake file for the install tree should not specify the ITK_DIR.
unset(ELASTIX_ITK_DIR)
configure_file(ElastixConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ElastixConfig.cmake" @ONLY)

configure_file(ElastixConfigVersion.cmake.in ElastixConfigVersion.cmake @ONLY)
configure_file(UseElastix.cmake.in UseElastix.cmake @ONLY)

if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ElastixConfig.cmake"
    "${PROJECT_BINARY_DIR}/ElastixConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/UseElastix.cmake"
    DESTINATION ${ELASTIX_INSTALL_PACKAGE_DIR}
    COMPONENT Development
    )

  install(EXPORT ElastixTargets
    DESTINATION ${ELASTIX_INSTALL_PACKAGE_DIR}
    COMPONENT Development
    )
endif()
