PROJECT(elxCore)

SET(KernelFiles 
Kernel/elxElastixBase.cxx
Kernel/elxElastixBase.h
Kernel/elxElastixMain.cxx
Kernel/elxElastixMain.h
Kernel/elxElastixTemplate.h
Kernel/elxElastixTemplate.hxx
Kernel/elxTransformixMain.cxx
Kernel/elxTransformixMain.h )

SET(InstallFiles
Install/elxBaseComponent.h
Install/elxBaseComponentSE.h
Install/elxBaseComponentSE.hxx
Install/elxComponentDatabase.cxx
Install/elxComponentDatabase.h
Install/elxComponentLoader.cxx
Install/elxComponentLoader.h
Install/elxInstallFunctions.h
Install/elxMacro.h
Install/elxIncludes.h
Install/elxSupportedImageTypes.h
Install/elxDynamicLoader.h
Install/elxDynamicLoader.cxx)

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
 
SOURCE_GROUP("Install" FILES ${InstallFiles} )
SOURCE_GROUP("Kernel" FILES ${KernelFiles} )
SOURCE_GROUP("Configuration" FILES ${ConfigurationFiles} )
SOURCE_GROUP("ComponentBaseClasses" FILES ${ComponentBaseClassFiles} )

ADD_LIBRARY(elxCore
${KernelFiles}
${ConfigurationFiles}
${ComponentBaseClassFiles} )

TARGET_LINK_LIBRARIES(elxCore
elxCommon
param
xoutlib
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )

ADD_EXECUTABLE(elastix Main/elastix.cxx Main/elastix.h)
ADD_EXECUTABLE(transformix Main/transformix.cxx Main/transformix.h)

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
