PROJECT(elxCore)


SET(KernelFilesForExecutables
 Kernel/elxElastixMain.cxx
 Kernel/elxElastixMain.h
 Kernel/elxTransformixMain.cxx
 Kernel/elxTransformixMain.h
)

SET(KernelFilesForComponents
 Kernel/elxElastixBase.cxx
 Kernel/elxElastixBase.h
 Kernel/elxElastixTemplate.h
 Kernel/elxElastixTemplate.hxx
)

SET(InstallFilesForExecutables
 Install/elxComponentLoader.cxx
 Install/elxComponentLoader.h
)

SET(InstallFilesForComponents
 Install/elxComponentDatabase.cxx
 Install/elxComponentDatabase.h
 Install/elxBaseComponent.h
 Install/elxBaseComponentSE.h
 Install/elxBaseComponentSE.hxx
 Install/elxInstallAllComponents.h
 Install/elxInstallFunctions.h
 Install/elxMacro.h
 Install/elxIncludes.h
 Install/elxSupportedImageTypes.h
)

SET(ConfigurationFiles
 Configuration/elxConfigurationBase.cxx
 Configuration/elxConfigurationBase.h
 Configuration/elxConfigurationToUse.h
 Configuration/elxMyConfiguration.cxx
 Configuration/elxMyConfiguration.h
)

SET(ComponentBaseClassFiles
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

SOURCE_GROUP("Install" FILES
 ${InstallFilesForComponents}
 ${InstallFilesForExecutables}
)
	
SOURCE_GROUP("Kernel" FILES
 ${KernelFilesForComponents}
 ${KernelFilesForExecutables}
)

SOURCE_GROUP( "Configuration" FILES ${ConfigurationFiles} )
SOURCE_GROUP( "ComponentBaseClasses" FILES ${ComponentBaseClassFiles} )
SOURCE_GROUP( "ProgressCommand" FILES ${ProgressCommandFiles} )

# on cygwin using gcc, -fPIC is ignored
IF ( NOT WIN32 )
  IF(CMAKE_COMPILER_IS_GNUCXX)
    #Add -fPIC to the gcc-options. 
    #This is needed for compilation on a 64-bit machine 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF(NOT WIN32)

ADD_LIBRARY(elxCore
 ${KernelFilesForComponents}
 ${InstallFilesForComponents}
 ${ConfigurationFiles}
 ${ComponentBaseClassFiles}
 ${ProgressCommandFiles}
)

TARGET_LINK_LIBRARIES(elxCore
 elxCommon
 xoutlib
 ITKBasicFilters
 ITKNumerics
 ITKIO
 ITKCommon
)
#param

ADD_EXECUTABLE(elastix
 Main/elastix.cxx
 Main/elastix.h
 Kernel/elxElastixMain.cxx
 Kernel/elxElastixMain.h
 ${InstallFilesForExecutables}
)

ADD_EXECUTABLE(transformix
 Main/transformix.cxx
 Main/transformix.h
 Kernel/elxElastixMain.cxx
 Kernel/elxElastixMain.h
 Kernel/elxTransformixMain.cxx
 Kernel/elxTransformixMain.h
 ${InstallFilesForExecutables}
)

TARGET_LINK_LIBRARIES(elastix
 elxCommon
 elxCore
 param
 xoutlib
 ITKBasicFilters
 ITKNumerics
 ITKIO
 ITKCommon
 ${AllComponentLibs}
)

TARGET_LINK_LIBRARIES(transformix
 elxCommon
 elxCore
 param
 xoutlib
 ITKBasicFilters
 ITKNumerics
 ITKIO
 ITKCommon
 ${AllComponentLibs}
)

INSTALL_TARGETS(/. elastix)
INSTALL_TARGETS(/. transformix)

