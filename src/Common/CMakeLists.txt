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
  itkAdvancedLinearInterpolateImageFunction.h
  itkAdvancedLinearInterpolateImageFunction.hxx
  itkAdvancedRayCastInterpolateImageFunction.h
  itkAdvancedRayCastInterpolateImageFunction.hxx
  itkGenericMultiResolutionPyramidImageFilter.h
  itkGenericMultiResolutionPyramidImageFilter.hxx
  itkImageFileCastWriter.h
  itkImageFileCastWriter.hxx
  itkMeshFileReaderBase.h
  itkMeshFileReaderBase.hxx
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.h
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.hxx
  itkMultiResolutionImageRegistrationMethod2.h
  itkMultiResolutionImageRegistrationMethod2.hxx
  itkMultiResolutionShrinkPyramidImageFilter.h
  itkMultiResolutionShrinkPyramidImageFilter.hxx
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
  itkImageMaskSpatialObject2.hxx
  itkImageSpatialObject2.h
  itkImageSpatialObject2.hxx
  itkErodeMaskImageFilter.h
  itkErodeMaskImageFilter.hxx
  itkParabolicErodeDilateImageFilter.h
  itkParabolicErodeDilateImageFilter.hxx
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
  CostFunctions/itkImageToImageMetricWithFeatures.hxx
  CostFunctions/itkLimiterFunctionBase.h
  CostFunctions/itkMultiInputImageToImageMetricBase.h
  CostFunctions/itkMultiInputImageToImageMetricBase.hxx
  CostFunctions/itkParzenWindowHistogramImageToImageMetric.h
  CostFunctions/itkParzenWindowHistogramImageToImageMetric.hxx
  CostFunctions/itkScaledSingleValuedCostFunction.cxx
  CostFunctions/itkScaledSingleValuedCostFunction.h
  CostFunctions/itkSingleValuedPointSetToPointSetMetric.h
  CostFunctions/itkSingleValuedPointSetToPointSetMetric.hxx
  CostFunctions/itkTransformPenaltyTerm.h
  CostFunctions/itkTransformPenaltyTerm.hxx
)

SET( TransformFiles
  Transforms/itkAdvancedBSplineDeformableTransformBase.h
  Transforms/itkAdvancedBSplineDeformableTransformBase.hxx
  Transforms/itkAdvancedBSplineDeformableTransform.h
  Transforms/itkAdvancedBSplineDeformableTransform.hxx
  Transforms/itkAdvancedCombinationTransform.h
  Transforms/itkAdvancedCombinationTransform.hxx
  Transforms/itkAdvancedIdentityTransform.h
  Transforms/itkAdvancedMatrixOffsetTransformBase.h
  Transforms/itkAdvancedMatrixOffsetTransformBase.hxx
  Transforms/itkAdvancedRigid2DTransform.h
  Transforms/itkAdvancedRigid2DTransform.hxx
  Transforms/itkAdvancedRigid3DTransform.h
  Transforms/itkAdvancedRigid3DTransform.hxx
  Transforms/itkAdvancedTransform.h
  Transforms/itkAdvancedTransform.hxx
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.hxx
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.hxx
  Transforms/itkBSplineInterpolationWeightFunction2.h
  Transforms/itkBSplineInterpolationWeightFunction2.hxx
  Transforms/itkBSplineInterpolationWeightFunctionBase.h
  Transforms/itkBSplineInterpolationWeightFunctionBase.hxx
  #Transforms/itkBSplineKernelFunction2.h
  #Transforms/itkBSplineSecondOrderDerivativeKernelFunction.h
  Transforms/itkBSplineSecondOrderDerivativeKernelFunction2.h
  Transforms/itkEulerTransform.h
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.h
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.hxx
  Transforms/itkTransformToSpatialJacobianSource.h
  Transforms/itkTransformToSpatialJacobianSource.hxx
  Transforms/itkGridScheduleComputer.h
  Transforms/itkGridScheduleComputer.hxx
  Transforms/itkUpsampleBSplineParametersFilter.h
  Transforms/itkUpsampleBSplineParametersFilter.hxx
)

SET( LineSearchOptimizersFiles
  LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.h
  LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.cxx
  LineSearchOptimizers/itkLineSearchOptimizer.h
  LineSearchOptimizers/itkLineSearchOptimizer.cxx
)

SET( ImageSamplersFiles
  ImageSamplers/itkImageFullSampler.h
  ImageSamplers/itkImageFullSampler.hxx
  ImageSamplers/itkImageGridSampler.h
  ImageSamplers/itkImageGridSampler.hxx
  ImageSamplers/itkImageRandomCoordinateSampler.h
  ImageSamplers/itkImageRandomCoordinateSampler.hxx
  ImageSamplers/itkImageRandomSampler.h
  ImageSamplers/itkImageRandomSampler.hxx
  ImageSamplers/itkImageRandomSamplerBase.h
  ImageSamplers/itkImageRandomSamplerSparseMask.h
  ImageSamplers/itkImageRandomSamplerSparseMask.hxx
  ImageSamplers/itkImageSample.h
  ImageSamplers/itkImageSamplerBase.h
  ImageSamplers/itkImageSamplerBase.hxx
  ImageSamplers/itkImageToVectorContainerFilter.h
  ImageSamplers/itkImageToVectorContainerFilter.hxx
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.h
  ImageSamplers/itkMultiInputImageRandomCoordinateSampler.hxx
  ImageSamplers/itkVectorContainerSource.h
  ImageSamplers/itkVectorContainerSource.hxx
  ImageSamplers/itkVectorDataContainer.h
  ImageSamplers/itkVectorDataContainer.hxx
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

