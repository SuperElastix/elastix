#---------------------------------------------------------------------
PROJECT( elxCore )

#---------------------------------------------------------------------
# Define lists of files in the subdirectories.

SET( KernelFilesForExecutables
  Kernel/elxElastixMain.cxx
  Kernel/elxElastixMain.h
  Kernel/elxTransformixMain.cxx
  Kernel/elxTransformixMain.h
)

SET( KernelFilesForComponents
  Kernel/elxElastixBase.cxx
  Kernel/elxElastixBase.h
  Kernel/elxElastixTemplate.h
  Kernel/elxElastixTemplate.hxx
)

SET( InstallFilesForExecutables
  Install/elxComponentLoader.cxx
  Install/elxComponentLoader.h
)

SET( InstallFilesForComponents
  Install/elxComponentDatabase.cxx
  Install/elxComponentDatabase.h
  Install/elxBaseComponent.h
  Install/elxBaseComponentSE.h
  Install/elxBaseComponentSE.hxx
  Install/elxInstallAllComponents.h
  Install/elxInstallFunctions.h
  Install/elxMacro.h
  Install/elxIncludes.h
)

SET( ConfigurationFiles
  Configuration/elxConfiguration.cxx
  Configuration/elxConfiguration.h
)

SET( ComponentBaseClassFiles
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

SET( ProgressCommandFiles
  elxProgressCommand.cxx
  elxProgressCommand.h
)

#---------------------------------------------------------------------
# Construct source groups for nice visualisation in Visual Studio.

SOURCE_GROUP( "Install" FILES
  ${InstallFilesForComponents}
  ${InstallFilesForExecutables}
)

SOURCE_GROUP( "Kernel" FILES
  ${KernelFilesForComponents}
  ${KernelFilesForExecutables}
)

SOURCE_GROUP( "Configuration" FILES ${ConfigurationFiles} )
SOURCE_GROUP( "ComponentBaseClasses" FILES ${ComponentBaseClassFiles} )
SOURCE_GROUP( "ProgressCommand" FILES ${ProgressCommandFiles} )

#---------------------------------------------------------------------
# Create the elxCore library.

ADD_LIBRARY( elxCore
  ${KernelFilesForComponents}
  ${InstallFilesForComponents}
  ${ConfigurationFiles}
  ${ComponentBaseClassFiles}
  ${ProgressCommandFiles}
)

#---------------------------------------------------------------------
# Link against other libraries.

TARGET_LINK_LIBRARIES( elxCore
  elxCommon
  xoutlib
  param # Needed for elxConfiguration
  #  ${ITK_LIBRARIES}
)

#---------------------------------------------------------------------
# Define the mevis dcm tiff lib to which we should link.
SET( mevisdcmtifflib mevisdcmtiff )

#---------------------------------------------------------------------
# Create the elastix executable or library.

IF( ELASTIX_BUILD_EXECUTABLE )
  ADD_EXECUTABLE( elastix
    Main/elastix.cxx
    Main/elastix.h
    Kernel/elxElastixMain.cxx
    Kernel/elxElastixMain.h
    ${InstallFilesForExecutables}
  )
ELSE(ELASTIX_BUILD_EXECUTABLE)
  IF( ELASTIX_BUILD_SHARED_LIBS )
    ADD_LIBRARY( elastix SHARED
      Main/elastixlib.cxx
      Main/elastixlib.h
      Kernel/elxElastixMain.cxx
      Kernel/elxElastixMain.h
      ${InstallFilesForExecutables}
    )
  ELSE(ELASTIX_BUILD_SHARED_LIBS)
    ADD_LIBRARY( elastix STATIC
      Main/elastixlib.cxx
      Main/elastixlib.h
      Kernel/elxElastixMain.cxx
      Kernel/elxElastixMain.h
      ${InstallFilesForExecutables}
    )
  ENDIF(ELASTIX_BUILD_SHARED_LIBS)
ENDIF(ELASTIX_BUILD_EXECUTABLE)


#---------------------------------------------------------------------
# Create the transformix executable.

ADD_EXECUTABLE( transformix
  Main/transformix.cxx
  Main/transformix.h
  Kernel/elxElastixMain.cxx
  Kernel/elxElastixMain.h
  Kernel/elxTransformixMain.cxx
  Kernel/elxTransformixMain.h
  ${InstallFilesForExecutables}
)

#---------------------------------------------------------------------
# Link elastix against other libraries.

TARGET_LINK_LIBRARIES( elastix
  param
  xoutlib
  elxCommon
  elxCore
  ${mevisdcmtifflib}
  ${AllComponentLibs}
  ${ITK_LIBRARIES}
)

#---------------------------------------------------------------------
# Link transformix against other libraries.

TARGET_LINK_LIBRARIES( transformix
  param
  xoutlib
  elxCommon
  elxCore
  ${mevisdcmtifflib}
  ${AllComponentLibs}
  ${ITK_LIBRARIES}
)

#---------------------------------------------------------------------
# Define the install directory for elastix and transformix.

IF( WIN32 )
  IF( ELASTIX_BUILD_EXECUTABLE )
    INSTALL( TARGETS elastix transformix
      RUNTIME DESTINATION . )
      #COMPONENT core )
  ELSE( ELASTIX_BUILD_EXECUTABLE )
    # transformix is not implemented as library yet
    INSTALL( TARGETS transformix
      RUNTIME DESTINATION . )
    INSTALL( TARGETS elastix 
      RUNTIME DESTINATION . 
      ARCHIVE DESTINATION . )
  ENDIF( ELASTIX_BUILD_EXECUTABLE )
ELSE( WIN32 )
  # Tell the executables where to find the required .so files.
  SET_TARGET_PROPERTIES( elastix transformix
    PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${ITK_DIR}" )
  # Put executables in bin and libraries in lib.
  INSTALL( TARGETS elastix transformix
    RUNTIME DESTINATION bin 
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION bin)
    #COMPONENT core )
ENDIF( WIN32 )

#---------------------------------------------------------------------
# The Core/Install directory contains a CMakeLists file for
# defining the elastix compilation types.

ADD_SUBDIRECTORY( Install )

