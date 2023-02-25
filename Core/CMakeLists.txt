#---------------------------------------------------------------------
project(elxCore)

#---------------------------------------------------------------------
# Define lists of files in the subdirectories.

set(KernelFilesForExecutables
  Kernel/elxMainBase.cxx
  Kernel/elxMainBase.h
  Kernel/elxElastixMain.cxx
  Kernel/elxElastixMain.h
  Kernel/elxTransformixMain.cxx
  Kernel/elxTransformixMain.h
)

set(KernelFilesForComponents
  Kernel/elxElastixBase.cxx
  Kernel/elxElastixBase.h
  Kernel/elxElastixTemplate.h
  Kernel/elxElastixTemplate.hxx
  Kernel/elxIterationInfo.cxx
  Kernel/elxIterationInfo.h
  Kernel/elxlog.h
  Kernel/elxlog.cxx
)

set(InstallFilesForExecutables
  Install/elxComponentLoader.cxx
  Install/elxComponentLoader.h
)

set(InstallFilesForComponents
  Install/elxComponentDatabase.cxx
  Install/elxComponentDatabase.h
  Install/elxComponentInstaller.h
  Install/elxConversion.cxx
  Install/elxConversion.h
  Install/elxBaseComponent.cxx
  Install/elxBaseComponent.h
  Install/elxBaseComponentSE.h
  Install/elxBaseComponentSE.hxx
  Install/elxInstallAllComponents.h
  Install/elxInstallFunctions.h
  Install/elxMacro.h
  Install/elxIncludes.h
)

set(ConfigurationFiles
  Configuration/elxConfiguration.cxx
  Configuration/elxConfiguration.h
)

set(ComponentBaseClassFiles
  ComponentBaseClasses/elxFixedImagePyramidBase.h
  ComponentBaseClasses/elxImageSamplerBase.h
  ComponentBaseClasses/elxInterpolatorBase.h
  ComponentBaseClasses/elxMetricBase.h
  ComponentBaseClasses/elxMovingImagePyramidBase.h
  ComponentBaseClasses/elxOptimizerBase.h
  ComponentBaseClasses/elxRegistrationBase.h
  ComponentBaseClasses/elxResampleInterpolatorBase.h
  ComponentBaseClasses/elxResamplerBase.h
  ComponentBaseClasses/elxTransformBase.h
  ComponentBaseClasses/elxFixedImagePyramidBase.hxx
  ComponentBaseClasses/elxImageSamplerBase.hxx
  ComponentBaseClasses/elxInterpolatorBase.hxx
  ComponentBaseClasses/elxMetricBase.hxx
  ComponentBaseClasses/elxMovingImagePyramidBase.hxx
  ComponentBaseClasses/elxOptimizerBase.hxx
  ComponentBaseClasses/elxRegistrationBase.hxx
  ComponentBaseClasses/elxResampleInterpolatorBase.hxx
  ComponentBaseClasses/elxResamplerBase.hxx
  ComponentBaseClasses/elxTransformBase.hxx
)

set(ProgressCommandFiles
  elxProgressCommand.cxx
  elxProgressCommand.h
)

#---------------------------------------------------------------------
# Construct source groups for nice visualisation in Visual Studio.

source_group("Install" FILES
  ${InstallFilesForComponents}
  ${InstallFilesForExecutables}
)

source_group("Kernel" FILES
  ${KernelFilesForComponents}
  ${KernelFilesForExecutables}
)

source_group("Configuration" FILES ${ConfigurationFiles})
source_group("ComponentBaseClasses" FILES ${ComponentBaseClassFiles})
source_group("ProgressCommand" FILES ${ProgressCommandFiles})

#---------------------------------------------------------------------

configure_file(
  elxVersionMacros.h.in
  elxVersionMacros.h)

if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/elxVersionMacros.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Core
    COMPONENT Development
    )
endif()
#---------------------------------------------------------------------

# Originally based on the answer by Naszta at
# https://stackoverflow.com/questions/6526451/how-to-include-git-commit-number-into-a-c-executable
find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE ELX_GIT_REVISION_DATE
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE ELX_GIT_REVISION_SHA
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

configure_file(elxGitRevisionInfo.h.in elxGitRevisionInfo.h @ONLY)

#---------------------------------------------------------------------
# Create the elxCore library.

add_library(elxCore STATIC
  ${KernelFilesForComponents}
  ${InstallFilesForComponents}
  ${ConfigurationFiles}
  ${ComponentBaseClassFiles}
  ${ProgressCommandFiles}
)

set_source_files_properties(Kernel/elxlog.cxx PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/ThirdParty/spdlog/include")

elastix_export_target(elxCore)

#---------------------------------------------------------------------
# Link against other libraries.

target_link_libraries(elxCore
  elxCommon
  param # Needed for elxConfiguration
  #  ${ITK_LIBRARIES}
)

# Support #include <double-conversion.h>, which is either in the root of the
# ITK install directory (its "install prefix"), or at the following location
# of the ITK source tree (which is the parent directory of "{ITK_CMAKE_DIR}"):
target_include_directories(elxCore PRIVATE
  ${ITK_CMAKE_DIR}/../Modules/ThirdParty/DoubleConversion/src/double-conversion)

elastix_export_target(elxCommon)
elastix_export_target(param)

#---------------------------------------------------------------------
# Define the mevis dcm tiff lib to which we should link.
set(mevisdcmtifflib mevisdcmtiff)

set(ELASTIX_TARGET_LINK_LIBRARIES
  param
  elxCommon
  elxCore
  ${mevisdcmtifflib}
  ${AllComponentLibs}
  ${ITK_LIBRARIES}
)
if(ELASTIX_USE_OPENCL)
  set(ELASTIX_TARGET_LINK_LIBRARIES  ${ELASTIX_TARGET_LINK_LIBRARIES} elxOpenCL)
endif()

#---------------------------------------------------------------------
# Create the elastix executable and library.
add_executable(elastix_exe
  Main/elastix.cxx
  Main/elastix.h
  Main/elxMainExeUtilities.cxx
  Main/elxMainExeUtilities.h
  Kernel/elxMainBase.cxx
  Kernel/elxMainBase.h
  Kernel/elxElastixMain.cxx
  Kernel/elxElastixMain.h
  ${InstallFilesForExecutables}
)
set_target_properties(elastix_exe PROPERTIES OUTPUT_NAME elastix)
target_compile_definitions(elastix_exe PRIVATE ELX_CMAKE_VERSION="${CMAKE_VERSION}")
target_link_libraries(elastix_exe ${ELASTIX_TARGET_LINK_LIBRARIES})

# The library type (STATIC or SHARED) is determined by the parameter
# BUILD_SHARED_LIBS.
add_library(elastix_lib
  Main/elxParameterObject.cxx
  Main/elxParameterObject.h
  Main/elastixlib.cxx
  Main/elastixlib.h
  Kernel/elxMainBase.cxx
  Kernel/elxMainBase.h
  Kernel/elxElastixMain.cxx
  Kernel/elxElastixMain.h
  ${InstallFilesForExecutables}
)
set_target_properties(elastix_lib PROPERTIES OUTPUT_NAME elastix)
target_compile_definitions(elastix_lib PRIVATE ELX_CMAKE_VERSION="${CMAKE_VERSION}")
target_link_libraries(elastix_lib ${ELASTIX_TARGET_LINK_LIBRARIES})

#---------------------------------------------------------------------
# Create the transformix executable.

add_executable(transformix_exe
  Main/transformix.cxx
  Main/elastix.h
  Main/elxMainExeUtilities.cxx
  Main/elxMainExeUtilities.h
  Kernel/elxMainBase.cxx
  Kernel/elxMainBase.h
  Kernel/elxTransformixMain.cxx
  Kernel/elxTransformixMain.h
  ${InstallFilesForExecutables}
)
set_target_properties(transformix_exe PROPERTIES OUTPUT_NAME transformix)
target_compile_definitions(transformix_exe PRIVATE ELX_CMAKE_VERSION="${CMAKE_VERSION}")
target_link_libraries(transformix_exe ${ELASTIX_TARGET_LINK_LIBRARIES})

# The library type (STATIC or SHARED) is determined by the parameter
# BUILD_SHARED_LIBS.
add_library(transformix_lib
  Main/elxParameterObject.cxx
  Main/elxParameterObject.h
  Main/transformixlib.cxx
  Main/transformixlib.h
  Kernel/elxMainBase.cxx
  Kernel/elxMainBase.h
  Kernel/elxTransformixMain.cxx
  Kernel/elxTransformixMain.h
  ${InstallFilesForExecutables}
)
set_target_properties(transformix_lib PROPERTIES OUTPUT_NAME transformix)
target_link_libraries(transformix_lib ${ELASTIX_TARGET_LINK_LIBRARIES})

set(ELASTIX_LIBRARIES elastix_lib transformix_lib PARENT_SCOPE)

#---------------------------------------------------------------------
# Define the install directory for elastix and transformix.

if(NOT WIN32)
  # Tell the executables where to find the required .so files.
  set_target_properties(elastix_exe transformix_exe
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${ITK_DIR}")
endif()

if(NOT ELASTIX_NO_INSTALL_EXECUTABLES)
  install(TARGETS elastix_exe transformix_exe
    ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
    LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
    RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR}
    COMPONENT Executables)
endif()

# Install all header files.
if(NOT ELASTIX_NO_INSTALL_DEVELOPMENT)
  install(DIRECTORY
    ${elastix_SOURCE_DIR}/Common
    ${elastix_SOURCE_DIR}/Core
    ${elastix_SOURCE_DIR}/Components
    DESTINATION ${ELASTIX_INCLUDE_DIR}
    COMPONENT Development
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hxx")
  install(FILES
    ${elastix_BINARY_DIR}/elxInstallComponentFunctionCalls.h
    ${elastix_BINARY_DIR}/elxInstallComponentFunctionDeclarations.h
    ${elastix_BINARY_DIR}/elxSupportedImageTypes.h
    ${elastix_BINARY_DIR}/ITKFactoryRegistration/itkImageIOFactoryRegisterManager.h
    ${elastix_BINARY_DIR}/ITKFactoryRegistration/itkTransformIOFactoryRegisterManager.h
    DESTINATION ${ELASTIX_INCLUDE_DIR}
    COMPONENT Development)
endif()

#---------------------------------------------------------------------
# The Core/Install directory contains a CMakeLists file for
# defining the elastix compilation types.

add_subdirectory(Install)

