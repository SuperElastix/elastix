#---------------------------------------------------------------------
project( elxCommon )

#---------------------------------------------------------------------
# Sub-projects.

add_subdirectory( ParameterFileParser )
add_subdirectory( xout )

if( USE_KNNGraphAlphaMutualInformationMetric )
  add_subdirectory( KNN )
endif()

add_subdirectory( MevisDicomTiff )

if( ELASTIX_USE_OPENCL )
  add_subdirectory( OpenCL )
endif()

#---------------------------------------------------------------------
# Define lists of files in the subdirectories.

set( CommonFiles
  itkAdvancedLinearInterpolateImageFunction.h
  itkAdvancedLinearInterpolateImageFunction.hxx
  itkAdvancedRayCastInterpolateImageFunction.h
  itkAdvancedRayCastInterpolateImageFunction.hxx
  itkComputeDisplacementDistribution.h
  itkComputeDisplacementDistribution.hxx
  itkComputeJacobianTerms.h
  itkComputeJacobianTerms.hxx
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
  TypeList.h
)

set( MaskFiles
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

set( CostFunctionFiles
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

set( TransformFiles
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

set( LineSearchOptimizersFiles
  LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.h
  LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.cxx
  LineSearchOptimizers/itkLineSearchOptimizer.h
  LineSearchOptimizers/itkLineSearchOptimizer.cxx
)

set( ImageSamplersFiles
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

source_group( "Masks" FILES ${MaskFiles} )
source_group( "CostFunctions" FILES ${CostFunctionFiles} )
source_group( "Transforms" FILES ${TransformFiles} )
source_group( "LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )
source_group( "ImageSamplers" FILES ${ImageSamplersFiles} )

#---------------------------------------------------------------------
# Create the elxCommon library.

add_library( elxCommon STATIC
  ${MaskFiles}
  ${CommonFiles}
  ${CostFunctionFiles}
  ${TransformFiles}
  ${LineSearchOptimizersFiles}
  ${ImageSamplersFiles}
)

#---------------------------------------------------------------------
# Link against other libraries.

if( UNIX AND NOT APPLE )
  target_link_libraries( elxCommon
    ${ITK_LIBRARIES}
    rt # Needed for elxTimer, clock_gettime()
  )
else()
  target_link_libraries( elxCommon
    ${ITK_LIBRARIES}
  )
endif()

