PROJECT(elxCore)


SET(KernelFilesForExecutables
Kernel/elxElastixMain.cxx
Kernel/elxElastixMain.h
Kernel/elxTransformixMain.cxx
Kernel/elxTransformixMain.h )

SET(KernelFilesForComponents
Kernel/elxElastixBase.cxx
Kernel/elxElastixBase.h
Kernel/elxElastixTemplate.h
Kernel/elxElastixTemplate.hxx)

SET(InstallFilesForExecutables
Install/elxComponentLoader.cxx
Install/elxComponentLoader.h
Install/elxDynamicLoader.h
Install/elxDynamicLoader.cxx)

SET(InstallFilesForComponents
Install/elxComponentDatabase.cxx
Install/elxComponentDatabase.h
Install/elxBaseComponent.h
Install/elxBaseComponentSE.h
Install/elxBaseComponentSE.hxx
Install/elxInstallFunctions.h
Install/elxMacro.h
Install/elxIncludes.h
Install/elxSupportedImageTypes.h)

SET(ConfigurationFiles
Configuration/elxConfigurationBase.cxx
Configuration/elxConfigurationBase.h
Configuration/elxConfigurationToUse.h
Configuration/elxMyConfiguration.cxx
Configuration/elxMyConfiguration.h )

SET(ComponentBaseClassFiles
ComponentBaseClasses/elxFixedImagePyramidBase.h
ComponentBaseClasses/elxInterpolatorBase.h
ComponentBaseClasses/elxMetricBase.h
ComponentBaseClasses/elxMovingImagePyramidBase.h
ComponentBaseClasses/elxOptimizerBase.h
ComponentBaseClasses/elxRegistrationBase.h
ComponentBaseClasses/elxResampleInterpolatorBase.h
ComponentBaseClasses/elxResamplerBase.h
ComponentBaseClasses/elxTransformBase.h
ComponentBaseClasses/elxFixedImagePyramidBase.hxx
ComponentBaseClasses/elxInterpolatorBase.hxx
ComponentBaseClasses/elxMetricBase.hxx
ComponentBaseClasses/elxMovingImagePyramidBase.hxx
ComponentBaseClasses/elxOptimizerBase.hxx
ComponentBaseClasses/elxRegistrationBase.hxx
ComponentBaseClasses/elxResampleInterpolatorBase.hxx
ComponentBaseClasses/elxResamplerBase.hxx
ComponentBaseClasses/elxTransformBase.hxx )
 

SOURCE_GROUP("Install" FILES
	${InstallFilesForComponents}
	${InstallFilesForExecutables} )
	
SOURCE_GROUP("Kernel" FILES
${KernelFilesForComponents}
${KernelFilesForExecutables} )

SOURCE_GROUP("Configuration" FILES ${ConfigurationFiles} )
SOURCE_GROUP("ComponentBaseClasses" FILES ${ComponentBaseClassFiles} )


ADD_LIBRARY(elxCore 
${KernelFilesForComponents}
${InstallFilesForComponents}
${ConfigurationFiles}
${ComponentBaseClassFiles} )

TARGET_LINK_LIBRARIES(elxCore
elxCommon
xoutlib
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )
#param

ADD_EXECUTABLE(elastix
Main/elastix.cxx
Main/elastix.h
Kernel/elxElastixMain.cxx
Kernel/elxElastixMain.h
${InstallFilesForExecutables} )

ADD_EXECUTABLE(transformix
Main/transformix.cxx
Main/transformix.h
Kernel/elxElastixMain.cxx
Kernel/elxElastixMain.h
Kernel/elxTransformixMain.cxx
Kernel/elxTransformixMain.h
${InstallFilesForExecutables} )

TARGET_LINK_LIBRARIES(elastix
elxCommon
elxCore
param
xoutlib
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )


TARGET_LINK_LIBRARIES(transformix
elxCommon
elxCore
param
xoutlib
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )

INSTALL_TARGETS(/. elastix)
INSTALL_TARGETS(/. transformix)
