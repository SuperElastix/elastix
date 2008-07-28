PROJECT( elxCommon )

SUBDIRS( KNN )
SUBDIRS( ParameterFileParser )
SUBDIRS( xout )

SET(CommonFiles
  elxTimer.cxx
  elxTimer.h
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
  itkImageFileCastWriter.h
  itkImageFileCastWriter.txx
  itkImageMaskSpatialObject2.h
  itkImageMaskSpatialObject2.txx
  itkImageSpatialObject2.h
  itkImageSpatialObject2.txx
  itkMeshFileReaderBase.h
  itkMeshFileReaderBase.txx
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.h
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.txx
  itkMultiResolutionImageRegistrationMethod2.h
  itkMultiResolutionImageRegistrationMethod2.txx
  itkTransformixInputPointFileReader.h
  itkTransformixInputPointFileReader.hxx
  itkNDImageBase.h
  itkNDImageTemplate.h
  itkNDImageTemplate.hxx
  itkScaledSingleValuedNonLinearOptimizer.cxx
  itkScaledSingleValuedNonLinearOptimizer.h
  itkTransformToDeformationFieldSource.h
  itkTransformToDeformationFieldSource.txx
  itkTranslationTransformInitializer.h
  itkTranslationTransformInitializer.hxx
)

SET( CostFunctionFiles
	CostFunctions/itkAdvancedImageToImageMetric.h
 	CostFunctions/itkAdvancedImageToImageMetric.hxx
	CostFunctions/itkExponentialLimiterFunction.h
  CostFunctions/itkExponentialLimiterFunction.hxx
  CostFunctions/itkHardLimiterFunction.h
  CostFunctions/itkHardLimiterFunction.hxx
  CostFunctions/itkImageToImageMetricWithFeatures.h
  CostFunctions/itkImageToImageMetricWithFeatures.txx
  CostFunctions/itkLimiterFunctionBase.h
  CostFunctions/itkMultiInputImageToImageMetricBase.h
  CostFunctions/itkMultiInputImageToImageMetricBase.txx
  CostFunctions/itkParzenWindowHistogramImageToImageMetric.h
  CostFunctions/itkParzenWindowHistogramImageToImageMetric.hxx
  CostFunctions/itkScaledSingleValuedCostFunction.cxx
  CostFunctions/itkScaledSingleValuedCostFunction.h
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
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.h
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.txx
  ImageSamplers/itkImageRandomSampler.h
  ImageSamplers/itkImageRandomSampler.txx
  ImageSamplers/itkImageRandomSamplerSparseMask.h
  ImageSamplers/itkImageRandomSamplerSparseMask.txx
  ImageSamplers/itkImageSample.h
  ImageSamplers/itkImageSamplerBase.h
  ImageSamplers/itkImageSamplerBase.txx
  ImageSamplers/itkImageRandomSamplerBase.h
  ImageSamplers/itkImageToVectorContainerFilter.h
  ImageSamplers/itkImageToVectorContainerFilter.txx
  ImageSamplers/itkVectorContainerSource.h
  ImageSamplers/itkVectorContainerSource.txx
  ImageSamplers/itkVectorDataContainer.h
  ImageSamplers/itkVectorDataContainer.txx
)

SOURCE_GROUP( "CostFunctions" FILES ${CostFunctionFiles} )
SOURCE_GROUP( "LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )
SOURCE_GROUP( "ImageSamplers" FILES ${ImageSamplersFiles} )

# on cygwin using gcc, -fPIC is ignored
IF(NOT WIN32)
  IF(CMAKE_COMPILER_IS_GNUCXX)
    #Add -fPIC to the gcc-options. 
    #This is needed for compilation on a 64-bit machine 
    #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF(NOT WIN32)

ADD_LIBRARY( elxCommon
  ${CommonFiles}
	${CostFunctionFiles}
  ${LineSearchOptimizersFiles}
  ${ImageSamplersFiles}
)

TARGET_LINK_LIBRARIES( elxCommon
  ITKBasicFilters
  ITKNumerics
  ITKIO
  ITKCommon
)

