PROJECT(elxCommon)

SUBDIRS(KNN)
SUBDIRS(ParameterFileParser)
SUBDIRS(xout)

SET(CommonFiles
 elxTimer.cxx
 elxTimer.h
 itkAdvancedImageToImageMetric.h
 itkAdvancedImageToImageMetric.hxx
 itkBSplineCombinationTransform.h
 itkBSplineCombinationTransform.hxx
 itkCombinationTransform.h
 itkCombinationTransform.hxx
 itkDeformationFieldInterpolatingTransform.h
 itkDeformationFieldInterpolatingTransform.txx
 itkDeformationFieldRegulizer.h
 itkDeformationFieldRegulizer.hxx
 itkDeformationFieldRegulizerForBSpline.h
 itkDeformationFieldRegulizerForBSpline.hxx
 itkDeformationVectorFieldTransform.h
 itkDeformationVectorFieldTransform.txx
 itkErodeMaskImageFilter.h
 itkErodeMaskImageFilter.txx
 itkExponentialLimiterFunction.h
 itkExponentialLimiterFunction.hxx
 itkImageFileCastWriter.h
 itkImageFileCastWriter.txx
 itkImageMaskSpatialObject2.h
 itkImageMaskSpatialObject2.txx
 itkImageSpatialObject2.h
 itkImageSpatialObject2.txx
 itkImageToImageMetricWithFeatures.h
 itkImageToImageMetricWithFeatures.txx
 itkHardLimiterFunction.h
 itkHardLimiterFunction.hxx
 itkLimiterFunctionBase.h
 itkMeshFileReaderBase.h
 itkMeshFileReaderBase.txx
 itkTransformixInputPointFileReader.h
 itkTransformixInputPointFileReader.hxx
 itkNDImageBase.h
 itkNDImageTemplate.h
 itkNDImageTemplate.hxx
 itkParzenWindowHistogramImageToImageMetric.h
 itkParzenWindowHistogramImageToImageMetric.hxx
 itkScaledSingleValuedCostFunction.cxx
 itkScaledSingleValuedCostFunction.h
 itkScaledSingleValuedNonLinearOptimizer.cxx
 itkScaledSingleValuedNonLinearOptimizer.h
 itkTranslationTransformInitializer.h
 itkTranslationTransformInitializer.hxx
)

SET( LineSearchOptimizersFiles
 LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.h
 LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.cxx
 LineSearchOptimizers/itkLineSearchOptimizer.h
 LineSearchOptimizers/itkLineSearchOptimizer.cxx
)

SET( ImageSamplersFiles
 ImageSamplers/itkImageFullSampler.h
 ImageSamplers/itkImageFullSampler.txx
 ImageSamplers/itkImageGridSampler.h
 ImageSamplers/itkImageGridSampler.txx
 ImageSamplers/itkImageRandomCoordinateSampler.h
 ImageSamplers/itkImageRandomCoordinateSampler.txx
 ImageSamplers/itkImageRandomSampler.h
 ImageSamplers/itkImageRandomSampler.txx
 ImageSamplers/itkImageRandomSamplerSparseMask.h
 ImageSamplers/itkImageRandomSamplerSparseMask.txx
 ImageSamplers/itkImageSample.h
 ImageSamplers/itkImageSamplerBase.h
 ImageSamplers/itkImageSamplerBase.txx
 ImageSamplers/itkImageToVectorContainerFilter.h
 ImageSamplers/itkImageToVectorContainerFilter.txx
 ImageSamplers/itkVectorContainerSource.h
 ImageSamplers/itkVectorContainerSource.txx
 ImageSamplers/itkVectorDataContainer.h
 ImageSamplers/itkVectorDataContainer.txx
)

SOURCE_GROUP("LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )
SOURCE_GROUP("ImageSamplers" FILES ${ImageSamplersFiles} )

IF(CMAKE_COMPILER_IS_GNUCXX)
  #Add -fPIC to the gcc-options. 
  #This is needed for compilation on a 64-bit machine 
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

ADD_LIBRARY(elxCommon
  ${ELX_GDCMHACK}
  ${CommonFiles}
	${LineSearchOptimizersFiles}
	${ImageSamplersFiles})

TARGET_LINK_LIBRARIES(elxCommon
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )

