#---------------------------------------------------------------------
PROJECT( elxCommon )

#---------------------------------------------------------------------
# Sub-projects.

ADD_SUBDIRECTORY( ParameterFileParser )
ADD_SUBDIRECTORY( xout )

IF( USE_KNNGraphAlphaMutualInformationMetric )
  ADD_SUBDIRECTORY( KNN )
ENDIF()

ADD_SUBDIRECTORY( MevisDicomTiff )

IF( ELASTIX_USE_CUDA )
  ADD_SUBDIRECTORY( CUDA )
ENDIF()

#---------------------------------------------------------------------
# Define lists of files in the subdirectories.

SET( CommonFiles
  elxTimer.cxx
  elxTimer.h
  itkAdvancedLinearInterpolateImageFunction.h
  itkAdvancedLinearInterpolateImageFunction.hxx
  itkAdvancedRayCastInterpolateImageFunction.h
  itkAdvancedRayCastInterpolateImageFunction.txx
  itkGenericMultiResolutionPyramidImageFilter.h
  itkGenericMultiResolutionPyramidImageFilter.hxx
  itkImageFileCastWriter.h
  itkImageFileCastWriter.txx
  itkMeshFileReaderBase.h
  itkMeshFileReaderBase.txx
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.h
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.txx
  itkMultiResolutionImageRegistrationMethod2.h
  itkMultiResolutionImageRegistrationMethod2.txx
  itkMultiResolutionShrinkPyramidImageFilter.h
  itkMultiResolutionShrinkPyramidImageFilter.txx
  itkNDImageBase.h
  itkNDImageTemplate.h
  itkNDImageTemplate.hxx
  itkScaledSingleValuedNonLinearOptimizer.cxx
  itkScaledSingleValuedNonLinearOptimizer.h
  itkTransformixInputPointFileReader.h
  itkTransformixInputPointFileReader.hxx
)

SET( MaskFiles
  itkImageMaskSpatialObject2.h
  itkImageMaskSpatialObject2.txx
  itkImageSpatialObject2.h
  itkImageSpatialObject2.txx
  itkErodeMaskImageFilter.h
  itkErodeMaskImageFilter.txx
  itkParabolicErodeDilateImageFilter.h
  itkParabolicErodeDilateImageFilter.txx
  itkParabolicErodeImageFilter.h
  itkParabolicMorphUtils.h
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
  CostFunctions/itkSingleValuedPointSetToPointSetMetric.h
  CostFunctions/itkSingleValuedPointSetToPointSetMetric.txx
  CostFunctions/itkTransformPenaltyTerm.h
  CostFunctions/itkTransformPenaltyTerm.txx
)

SET( TransformFiles
  Transforms/itkAdvancedBSplineDeformableTransformBase.h
  Transforms/itkAdvancedBSplineDeformableTransformBase.txx
  Transforms/itkAdvancedBSplineDeformableTransform.h
  Transforms/itkAdvancedBSplineDeformableTransform.txx
  Transforms/itkAdvancedCombinationTransform.h
  Transforms/itkAdvancedCombinationTransform.hxx
  Transforms/itkAdvancedIdentityTransform.h
  Transforms/itkAdvancedMatrixOffsetTransformBase.h
  Transforms/itkAdvancedMatrixOffsetTransformBase.txx
  Transforms/itkAdvancedRigid2DTransform.h
  Transforms/itkAdvancedRigid2DTransform.txx
  Transforms/itkAdvancedRigid3DTransform.h
  Transforms/itkAdvancedRigid3DTransform.txx
  Transforms/itkAdvancedTransform.h
  Transforms/itkAdvancedTransform.txx
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.txx
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.txx
  Transforms/itkBSplineInterpolationWeightFunction2.h
  Transforms/itkBSplineInterpolationWeightFunction2.txx
  Transforms/itkBSplineInterpolationWeightFunctionBase.h
  Transforms/itkBSplineInterpolationWeightFunctionBase.txx
  #Transforms/itkBSplineKernelFunction2.h
  #Transforms/itkBSplineSecondOrderDerivativeKernelFunction.h
  Transforms/itkBSplineSecondOrderDerivativeKernelFunction2.h
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.h
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.txx
  Transforms/itkTransformToSpatialJacobianSource.h
  Transforms/itkTransformToSpatialJacobianSource.txx
  Transforms/itkGridScheduleComputer.h
  Transforms/itkGridScheduleComputer.txx
  Transforms/itkUpsampleBSplineParametersFilter.h
  Transforms/itkUpsampleBSplineParametersFilter.txx
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
  ImageSamplers/itkImageRandomSamplerBase.h
  ImageSamplers/itkImageRandomSamplerSparseMask.h
  ImageSamplers/itkImageRandomSamplerSparseMask.txx
  ImageSamplers/itkImageSample.h
  ImageSamplers/itkImageSamplerBase.h
  ImageSamplers/itkImageSamplerBase.txx
  ImageSamplers/itkImageToVectorContainerFilter.h
  ImageSamplers/itkImageToVectorContainerFilter.txx
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.h
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.txx
  ImageSamplers/itkVectorContainerSource.h
  ImageSamplers/itkVectorContainerSource.txx
  ImageSamplers/itkVectorDataContainer.h
  ImageSamplers/itkVectorDataContainer.txx
)

#---------------------------------------------------------------------
# Construct source groups for nice visualisation in Visual Studio.

SOURCE_GROUP( "Masks" FILES ${MaskFiles} )
SOURCE_GROUP( "CostFunctions" FILES ${CostFunctionFiles} )
SOURCE_GROUP( "Transforms" FILES ${TransformFiles} )
SOURCE_GROUP( "LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )
SOURCE_GROUP( "ImageSamplers" FILES ${ImageSamplersFiles} )

#---------------------------------------------------------------------
# Create the elxCommon library.

ADD_LIBRARY( elxCommon
  ${MaskFiles}
  ${CommonFiles}
  ${CostFunctionFiles}
  ${TransformFiles}
  ${LineSearchOptimizersFiles}
  ${ImageSamplersFiles}
)

#---------------------------------------------------------------------
# Link against other libraries.

IF( UNIX AND NOT APPLE )
  TARGET_LINK_LIBRARIES( elxCommon
    ${ITK_LIBRARIES}
    rt # Needed for elxTimer, clock_gettime()
  )
ELSE()
  TARGET_LINK_LIBRARIES( elxCommon
    ${ITK_LIBRARIES}
  )
ENDIF()

