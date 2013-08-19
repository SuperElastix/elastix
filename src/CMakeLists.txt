#---------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
CMAKE_POLICY( SET CMP0012 NEW )

#---------------------------------------------------------------------
PROJECT( elastix )

#---------------------------------------------------------------------
# Generate executable, static library or dynamic library.
MARK_AS_ADVANCED( ELASTIX_BUILD_EXECUTABLE )
OPTION( ELASTIX_BUILD_EXECUTABLE "Generate executable or library?" ON )

IF( NOT ELASTIX_BUILD_EXECUTABLE )
  ADD_DEFINITIONS( -D_ELASTIX_BUILD_LIBRARY )
  MARK_AS_ADVANCED( ELASTIX_BUILD_SHARED_LIBS )
  OPTION( ELASTIX_BUILD_SHARED_LIBS "Build shared libraries?" ON )
  IF( ELASTIX_BUILD_SHARED_LIBS )
    ADD_DEFINITIONS( -D_ELASTIX_BUILD_SHARED_LIBRARY )
  ENDIF()
ENDIF()


#---------------------------------------------------------------------
# Find ITK.
FIND_PACKAGE( ITK REQUIRED )
IF( ITK_FOUND )
  INCLUDE( ${ITK_USE_FILE} )
ENDIF( ITK_FOUND )

#---------------------------------------------------------------------
# Find CUDA.
MARK_AS_ADVANCED( ELASTIX_USE_CUDA )
OPTION( ELASTIX_USE_CUDA "Use CUDA enabled GPU" OFF )

IF( ELASTIX_USE_CUDA )
  FIND_PACKAGE( CUDA REQUIRED )

  MARK_AS_ADVANCED( CUDA_BUILD_CUBIN )
  MARK_AS_ADVANCED( CUDA_BUILD_EMULATION )
  MARK_AS_ADVANCED( CUDA_VERBOSE_BUILD )
  MARK_AS_ADVANCED( CUDA_SDK_ROOT_DIR )
ENDIF()

#---------------------------------------------------------------------
# Check for the SuiteSparse package
# We need to do that here, because the link_directories should be set
# before declaring any targets.

IF( USE_PreconditionedGradientDescent )
 SET( SUITESPARSE_BINARY_DIR "" CACHE PATH "Path to SuiteSparse" )
 SET( SUITESPARSE_USE_FILE ${SUITESPARSE_BINARY_DIR}/UseSuiteSparse.cmake )
 IF( EXISTS ${SUITESPARSE_USE_FILE} )
  INCLUDE( ${SUITESPARSE_USE_FILE} )
 ENDIF()
ENDIF()

#---------------------------------------------------------------------
# Set single output directories for all executables and libraries.
# This makes it easier to create an elxUseFile.cmake, that users can
# include in their programs to borrow elastix functionality.

IF( NOT LIBRARY_OUTPUT_PATH )
  SET( LIBRARY_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL
    "Single output directory for building all libraries." )
ENDIF()

IF( NOT EXECUTABLE_OUTPUT_PATH )
  SET( EXECUTABLE_OUTPUT_PATH ${elastix_BINARY_DIR}/bin CACHE INTERNAL
    "Single output directory for building all executables." )
ENDIF()

# Mark these variables as advanced; their default value is usually fine
MARK_AS_ADVANCED( LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH )

#---------------------------------------------------------------------
# Check if Mevis DicomTiff support is desired
MARK_AS_ADVANCED( ELASTIX_USE_MEVISDICOMTIFF )
OPTION( ELASTIX_USE_MEVISDICOMTIFF
  "Support MevisLab DicomTiff image format" OFF )

#---------------------------------------------------------------------
# Define cmake variable to define extra user component directories
# These directories will be added to the list of include directories
# and they will be searched for CMakeLists.txt files for further
# processing. In these directories, users may put code for their own
# components.

MARK_AS_ADVANCED( ELASTIX_USER_COMPONENT_DIRS )
SET( ELASTIX_USER_COMPONENT_DIRS "" CACHE PATH
  "directories with user defined elastix components" )

#---------------------------------------------------------------------
# elastix depends on some ITK settings

IF( NOT Module_ITKReview )
  message( SEND_ERROR "Module_ITKReview has not been set to ON when compiling ITK. This option is mandatory for elastix to build.\nPlease recompile your ITK installation." )
ENDIF()

#---------------------------------------------------------------------
# If IDE supports it, do use folder view.
# gcc automatically ignores it. VC Express does not support and gives
# annoying warnings when this option is turned on.
# VC pro does support it.
SET( CMAKE_USE_FOLDERS ON CACHE INTERNAL "Use folder view in IDE" )
IF( CMAKE_MAKE_PROGRAM MATCHES ".?VCExpress.?" )
  SET( CMAKE_USE_FOLDERS OFF CACHE INTERNAL "Use folder view in IDE" )
ENDIF()
SET_PROPERTY( GLOBAL PROPERTY USE_FOLDERS ${CMAKE_USE_FOLDERS} )

#---------------------------------------------------------------------
# Set default build type to Release, if none was specified
# Taken from ITK CMake list
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
  message( STATUS "Setting build type to 'Release' as none was specified." )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
endif()

#---------------------------------------------------------------------
# Include directories

SET( elxCommon_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Common
  ${elastix_SOURCE_DIR}/Common/CostFunctions
  ${elastix_SOURCE_DIR}/Common/CUDA
  ${elastix_SOURCE_DIR}/Common/ImageSamplers
  ${elastix_SOURCE_DIR}/Common/KNN
  ${elastix_SOURCE_DIR}/Common/KNN/ann_1.1/include
  ${elastix_SOURCE_DIR}/Common/LineSearchOptimizers
  ${elastix_SOURCE_DIR}/Common/ParameterFileParser
  ${elastix_SOURCE_DIR}/Common/Transforms
  ${elastix_SOURCE_DIR}/Common/xout
  ${elastix_SOURCE_DIR}/Common/MevisDicomTiff
)

SET( elxCore_INCLUDE_DIRECTORIES
  ${elastix_SOURCE_DIR}/Core
  ${elastix_SOURCE_DIR}/Core/Install
  ${elastix_SOURCE_DIR}/Core/Kernel
  ${elastix_SOURCE_DIR}/Core/ComponentBaseClasses
  ${elastix_SOURCE_DIR}/Core/Configuration
  ${elastix_SOURCE_DIR}/Core/Main
)

SET( elxComponents_INCLUDE_DIRECTORIES
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

SET( elxINCLUDE_DIRECTORIES
  ${elxCommon_INCLUDE_DIRECTORIES}
  ${elxCore_INCLUDE_DIRECTORIES}
  ${elxComponents_INCLUDE_DIRECTORIES}
  ${elastix_BINARY_DIR}
  ${ELASTIX_USER_COMPONENT_DIRS}
)

INCLUDE_DIRECTORIES( ${elxINCLUDE_DIRECTORIES} )

#---------------------------------------------------------------------
# Microsoft specific items

IF( MSVC )
  # Kill the anoying MS VS warning about non-safe functions.
  # They hide real warnings.
  ADD_DEFINITIONS(
    /D_SCL_SECURE_NO_DEPRECATE
    /D_CRT_SECURE_NO_DEPRECATE
    /D_CRT_TIME_FUNCTIONS_NO_DEPRECATE
  )

  # Increases address capacity
  IF ( WIN32 )
    SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj" )
    SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj" )
    SET( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /bigobj" )
  ENDIF()
ENDIF()

#---------------------------------------------------------------------
# Process the sub-directories

# Common dir: code that is neither related to the core of elastix or
# to specific components.
ADD_SUBDIRECTORY( Common )

# Components: the registration components such as metrics, transforms,
# optimizers, etc.
ADD_SUBDIRECTORY( Components )

# Core dir: code that takes care of starting elastix, loading
# components, definitions of macros etc.
ADD_SUBDIRECTORY( Core )

#---------------------------------------------------------------------
# Configure the examples

SET( ELASTIX_DOX_DIR  ${elastix_SOURCE_DIR}/../dox )
SET( ELASTIX_HELP_DIR ${elastix_SOURCE_DIR}/../help CACHE PATH
  "path to the doxygen generated help files and the examples" )

# Copy the examples to the help directory

IF( WIN32 )
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/example.bat
    ${ELASTIX_HELP_DIR}/example.bat
    COPYONLY )
ELSE()
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/example
    ${ELASTIX_HELP_DIR}/example
    COPYONLY )
ENDIF()

MAKE_DIRECTORY( ${ELASTIX_HELP_DIR}/exampleinput )
MAKE_DIRECTORY( ${ELASTIX_HELP_DIR}/exampleoutput )

SET( ELX_EXAMPLEINPUTFILES
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

FOREACH( ELX_EXAMPLEINPUTFILE ${ELX_EXAMPLEINPUTFILES} )
  CONFIGURE_FILE(
    ${ELASTIX_DOX_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    ${ELASTIX_HELP_DIR}/exampleinput/${ELX_EXAMPLEINPUTFILE}
    COPYONLY )
ENDFOREACH()

#---------------------------------------------------------------------
# Get version information.

# Get the version number of elastix
FILE( STRINGS ${elastix_SOURCE_DIR}/Core/Install/elxBaseComponent.h
  ELASTIX_VERSION REGEX "\(#define\ __ELASTIX_VERSION\)" )
STRING( SUBSTRING ${ELASTIX_VERSION} 26 3 ELASTIX_VERSION )

# Split version in major minor. Assuming no patch number, just x.y
STRING( REGEX MATCH "[0-9]+" ELASTIX_VERSION_MAJOR "${ELASTIX_VERSION}" )
STRING( REGEX REPLACE "([0-9]+)\\." "" ELASTIX_VERSION_MINOR "${ELASTIX_VERSION}" )

#---------------------------------------------------------------------
# Configure the doxygen-configuration

FIND_PACKAGE( Doxygen QUIET )
STRING( COMPARE NOTEQUAL ${DOXYGEN} "DOXYGEN-NOTFOUND" Doxygen_FOUND )

IF( Doxygen_FOUND )

  # Set the path to the doxygen configuration source
  SET( ELASTIX_DOXYGEN_DIR ${ELASTIX_DOX_DIR}/doxygen )

  # Get the version number of doxygen
  EXEC_PROGRAM( ${DOXYGEN} ARGS "--version" OUTPUT_VARIABLE ELASTIX_DOXYGEN_VERSION )

  # Get date
  IF( UNIX OR CYGWIN )
    EXEC_PROGRAM( "date '+%d-%m-%Y'" OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATE )
  ENDIF()
  IF( WIN32 )
    IF( NOT CYGWIN )
      EXECUTE_PROCESS( COMMAND "${ELASTIX_DOXYGEN_DIR}/doxdate.bat"
        OUTPUT_VARIABLE ELASTIX_DOXYGEN_DATETEMP )
      STRING( REPLACE "/" "-" ELASTIX_DOXYGEN_DATE ${ELASTIX_DOXYGEN_DATETEMP} )
    ENDIF()
  ENDIF()

  # Configure the doxygen configuration
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/doxyfile.in
    ${ELASTIX_HELP_DIR}/doxyfile.out @ONLY )

  # Configure the footer of the help website.
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/DoxygenFooter.html.in
    ${ELASTIX_HELP_DIR}/DoxygenFooter.html @ONLY )

  # Configure the MainPage.dox
  CONFIGURE_FILE(
    ${ELASTIX_DOXYGEN_DIR}/MainPage.dox.in
    ${ELASTIX_HELP_DIR}/MainPage.dox @ONLY )

ENDIF()

#---------------------------------------------------------------------
# Testing

SET( ELASTIX_BUILD_TESTING OFF CACHE BOOL
  "Perform some tests on basic functionality of elastix." )
IF( ELASTIX_BUILD_TESTING )
  ENABLE_TESTING()
  ADD_SUBDIRECTORY( Testing )
  INCLUDE( CTest )
ENDIF()

#---------------------------------------------------------------------
# Packaging

MARK_AS_ADVANCED( ELASTIX_ENABLE_PACKAGER )
OPTION( ELASTIX_ENABLE_PACKAGER "Enable elastix packager using CPack" OFF )

IF( ELASTIX_ENABLE_PACKAGER )
  # Version information
  # If the next line is uncommented the package name will be like
  # elastix-4.3-win64 instead of elastix-4.3.0-win64
  #SET( CPACK_PACKAGE_VERSION ${ELASTIX_VERSION} )
  SET( CPACK_PACKAGE_VERSION_MAJOR ${ELASTIX_VERSION_MAJOR} )
  SET( CPACK_PACKAGE_VERSION_MINOR ${ELASTIX_VERSION_MINOR} )
  SET( CPACK_PACKAGE_VERSION_PATCH "0" )

  # Also install the copyright file, since when the user enables packaging
  # we assume that the package is meant to distribute.
  INSTALL( FILES "${elastix_SOURCE_DIR}/CopyrightElastix.txt" DESTINATION . )

  # We have split the elastix package into two components:
  # - the core: elastix and transformix
  # - the libraries: ANNlib and cudart when ELASTIX_USE_CUDA ON
  # NOTE: Currently does not work for nsis. A bug related to this
  # seems to be fixed in the upcoming CMake 2.8.3
  # Therefore, disable component support for now.
  #SET( CPACK_COMPONENTS_ALL core libraries Unspecified )
  #  CPACK_ADD_COMPONENT( core
  #  "Core files"
  #  DESCRIPTION "Contains elastix and transformix"
  #  REQUIRED
  #  DEPENDS libraries )
#   [GROUP group]
#   [DEPENDS comp1 comp2 ... ]
#   [INSTALL_TYPES type1 type2 ... ]
#   [DOWNLOADED]
#   [ARCHIVE_FILE filename])
  #CPACK_ADD_COMPONENT( libraries
  #  "Libraries"
  #  DESCRIPTION "Contains the libraries"
  #  REQUIRED )
#SET( CPACK_COMPONENT_CORE_DISPLAY_NAME "Core files" )
#SET( CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries" )
#SET( CPACK_COMPONENT_CORE_DESCRIPTION "Contains elastix and transformix" )
#SET( CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Contains the libraries" )
#SET( CPACK_COMPONENT_CORE_DEPENDS libraries )
#SET( CPACK_COMPONENT_CORE_REQUIRED ON )
#SET( CPACK_COMPONENT_LIBRARIES_REQUIRED ON )

  # The default package filename is
  # ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}
  # which is like elastix-4.3.0-win64, or elastix-4.3.0-linux-i686
  # Currently however we use elastix_windows64_v4.3
  # We can change our naming schedule or come close to it using:
  #SET( CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_SYSTEM_NAME}_v${CPACK_PACKAGE_VERSION}" )
  # but it doesn't work since these variables are not defined yet
  # Moving include(CPack) to above introduces other errors.
  #SET( CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}_${CMAKE_SYSTEM_NAME}_v${ELASTIX_VERSION}" )
  # also does not work properly. Just use the default for now.

  # General information
  SET( CPACK_PACKAGE_VENDOR "Stefan Klein and Marius Staring" )
  SET( CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "elastix is an image registration toolkit" )
  #SET( CPACK_PACKAGE_DESCRIPTION_FILE
  #  "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt" )
  SET( CPACK_RESOURCE_FILE_LICENSE
    "${CMAKE_CURRENT_SOURCE_DIR}/CopyrightElastix.txt" )

  # The default install directories: .../elastix_v4.3
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY
    "elastix_v${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR}" )
  SET( CPACK_PACKAGE_INSTALL_REGISTRY_KEY
    "elastix_v${ELASTIX_VERSION_MAJOR}.${ELASTIX_VERSION_MINOR}" )

  # Do not include a subdirectory in the zip
  SET( CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0 )

  SET( CPACK_SET_DESTDIR "OFF")

  # Make sure the directory structure is kept in the source zip
  # and that also the dox directory is included.
  SET( CPACK_SOURCE_INSTALLED_DIRECTORIES
    "${elastix_SOURCE_DIR};/src;${ELASTIX_DOX_DIR};/dox" )

  # ??
  #SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")

  # For the windows nsis installer only (is this if-check necessary?)
  IF( WIN32 AND NOT UNIX )
    # NOTE: There is a bug in NSI that does not handle full unix paths properly.
    # Make sure there is at least one set of four (4) backlashes
    # (CMake escapes 2, and the other gets escaped too in some second step)

    # Set the generators. If left blank the user has all options.
    SET( CPACK_GENERATOR "NSIS;ZIP")

    # Adding information
    SET( CPACK_NSIS_DISPLAY_NAME
      "${CPACK_PACKAGE_INSTALL_DIRECTORY} elastix" )
    SET( CPACK_NSIS_HELP_LINK "http:\\\\\\\\elastix.isi.uu.nl" )
    SET( CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\elastix.isi.uu.nl" )
    SET( CPACK_NSIS_CONTACT "elastix@bigr.nl" )
    SET( CPACK_NSIS_PACKAGE_NAME "elastix" )
    SET( CPACK_NSIS_DISPLAY_NAME "elastix" )

    # Adding icons and images to make it look nice:
    # 1 A branding image that will be displayed inside the installer
    # 2 The icon file (.ico) for the generated install program
    # 3 The icon file (.ico) for the generated uninstall program
    SET( CPACK_PACKAGE_ICON
      "${CMAKE_CURRENT_SOURCE_DIR}/../dox/art\\\\elastix_logo_full_small.bmp" )
    SET( CPACK_NSIS_MUI_ICON
      "${CMAKE_CURRENT_SOURCE_DIR}/../dox/art\\\\elastix_logo_64.ico" )
    SET( CPACK_NSIS_MUI_UNIICON
      "${CMAKE_CURRENT_SOURCE_DIR}/../dox/art\\\\elastix_logo_64.ico" )

    # Create an option in the installer that asks if elastix
    # needs to be added to the system path
    SET( CPACK_NSIS_MODIFY_PATH ON )

  ELSE()
    # Set the generators
    SET( CPACK_GENERATOR "TBZ2;ZIP" )

    # SET(CPACK_STRIP_FILES "bin/MyExecutable")
    #SET(CPACK_SOURCE_STRIP_FILES "")
  ENDIF()

  # Uncomment the following line if we want to include the system
  # dll's in the distribution!
  #INCLUDE( InstallRequiredSystemLibraries )

  # This include will do all the work.
  INCLUDE( CPack )
ENDIF()

#---------------------------------------------------------------------
# Make it easier to include elastix functionality in other programs.

# Save library dependencies.
# The library dependencies file.
SET( elxLIBRARY_DEPENDS_FILE ${elastix_BINARY_DIR}/elxLibraryDepends.cmake )
EXPORT_LIBRARY_DEPENDENCIES( ${elxLIBRARY_DEPENDS_FILE} )

# The "use" file.
SET( elxUSE_FILE ${elastix_BINARY_DIR}/UseElastix.cmake )

# The build settings file. (necessary for elastix?)
#SET( ITK_BUILD_SETTINGS_FILE ${ITK_BINARY_DIR}/ITKBuildSettings.cmake )

# Create the use file.
CONFIGURE_FILE( ${elastix_SOURCE_DIR}/UseElastix.cmake.in
 ${elxUSE_FILE} @ONLY )

