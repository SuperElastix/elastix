#---------------------------------------------------------------------
project( elxCommon )

#---------------------------------------------------------------------
# Sub-projects.

add_subdirectory( ParameterFileParser )
add_subdirectory( xout )
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
  itkComputeImageExtremaFilter.h
  itkComputeImageExtremaFilter.hxx
  itkComputeDisplacementDistribution.h
  itkComputeDisplacementDistribution.hxx
  itkComputeJacobianTerms.h
  itkComputeJacobianTerms.hxx
  itkErodeMaskImageFilter.h
  itkErodeMaskImageFilter.hxx
  itkGenericMultiResolutionPyramidImageFilter.h
  itkGenericMultiResolutionPyramidImageFilter.hxx
  itkImageFileCastWriter.h
  itkImageFileCastWriter.hxx
  itkImageMaskSpatialObject2.h
  itkImageMaskSpatialObject2.hxx
  itkImageSpatialObject2.h
  itkImageSpatialObject2.hxx
  itkMeshFileReaderBase.h
  itkMeshFileReaderBase.hxx
  itkMultiOrderBSplineDecompositionImageFilter.h
  itkMultiOrderBSplineDecompositionImageFilter.hxx
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.h
  itkMultiResolutionGaussianSmoothingPyramidImageFilter.hxx
  itkMultiResolutionImageRegistrationMethod2.h
  itkMultiResolutionImageRegistrationMethod2.hxx
  itkMultiResolutionShrinkPyramidImageFilter.h
  itkMultiResolutionShrinkPyramidImageFilter.hxx
  itkNDImageBase.h
  itkNDImageTemplate.h
  itkNDImageTemplate.hxx
  itkParabolicErodeDilateImageFilter.h
  itkParabolicErodeDilateImageFilter.hxx
  itkParabolicErodeImageFilter.h
  itkParabolicMorphUtils.h
  itkRecursiveBSplineInterpolationWeightFunction.h
  itkRecursiveBSplineInterpolationWeightFunction.hxx
  itkReducedDimensionBSplineInterpolateImageFunction.h
  itkReducedDimensionBSplineInterpolateImageFunction.hxx
  itkScaledSingleValuedNonLinearOptimizer.cxx
  itkScaledSingleValuedNonLinearOptimizer.h
  itkTransformixInputPointFileReader.h
  itkTransformixInputPointFileReader.hxx
  TypeList.h
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
  Transforms/itkAdvancedEuler3DTransform.h
  Transforms/itkAdvancedEuler3DTransform.hxx
  Transforms/itkAdvancedIdentityTransform.h
  Transforms/itkAdvancedImageMomentsCalculator.h
  Transforms/itkAdvancedImageMomentsCalculator.hxx
  Transforms/itkAdvancedMatrixOffsetTransformBase.h
  Transforms/itkAdvancedMatrixOffsetTransformBase.hxx
  Transforms/itkAdvancedRigid2DTransform.h
  Transforms/itkAdvancedRigid2DTransform.hxx
  Transforms/itkAdvancedRigid3DTransform.h
  Transforms/itkAdvancedRigid3DTransform.hxx
  Transforms/itkAdvancedSimilarity2DTransform.h
  Transforms/itkAdvancedSimilarity2DTransform.hxx
  Transforms/itkAdvancedSimilarity3DTransform.h
  Transforms/itkAdvancedSimilarity3DTransform.hxx
  Transforms/itkAdvancedTransform.h
  Transforms/itkAdvancedTransform.hxx
  Transforms/itkAdvancedTranslationTransform.h
  Transforms/itkAdvancedTranslationTransform.hxx
  Transforms/itkAdvancedVersorTransform.h
  Transforms/itkAdvancedVersorTransform.hxx
  Transforms/itkAdvancedVersorRigid3DTransform.h
  Transforms/itkAdvancedVersorRigid3DTransform.hxx
  Transforms/itkBSplineDerivativeKernelFunction2.h
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationDerivativeWeightFunction.hxx
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h
  Transforms/itkBSplineInterpolationSecondOrderDerivativeWeightFunction.hxx
  Transforms/itkBSplineInterpolationWeightFunction2.h
  Transforms/itkBSplineInterpolationWeightFunction2.hxx
  Transforms/itkBSplineInterpolationWeightFunctionBase.h
  Transforms/itkBSplineInterpolationWeightFunctionBase.hxx
  Transforms/itkBSplineKernelFunction2.h
  Transforms/itkBSplineSecondOrderDerivativeKernelFunction2.h
  Transforms/itkCyclicBSplineDeformableTransform.h
  Transforms/itkCyclicBSplineDeformableTransform.hxx
  Transforms/itkCyclicGridScheduleComputer.h
  Transforms/itkCyclicGridScheduleComputer.hxx
  Transforms/itkEulerTransform.h
  Transforms/itkGridScheduleComputer.h
  Transforms/itkGridScheduleComputer.hxx
  Transforms/itkRecursiveBSplineTransform.hxx
  Transforms/itkRecursiveBSplineTransform.h
  Transforms/itkRecursiveBSplineTransformImplementation.h
  Transforms/itkStackTransform.h
  Transforms/itkStackTransform.hxx
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.h
  Transforms/itkTransformToDeterminantOfSpatialJacobianSource.hxx
  Transforms/itkTransformToSpatialJacobianSource.h
  Transforms/itkTransformToSpatialJacobianSource.hxx
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

source_group( "Common" FILES ${CommonFiles} )
source_group( "CostFunctions" FILES ${CostFunctionFiles} )
source_group( "Transforms" FILES ${TransformFiles} )
source_group( "LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )
source_group( "ImageSamplers" FILES ${ImageSamplersFiles} )

#---------------------------------------------------------------------
# Create the elxCommon library.

add_library( elxCommon STATIC
  ${CommonFiles}
  ${CostFunctionFiles}
  ${TransformFiles}
  ${LineSearchOptimizersFiles}
  ${ImageSamplersFiles}
)
install( TARGETS elxCommon
  ARCHIVE DESTINATION ${ELASTIX_ARCHIVE_DIR}
  LIBRARY DESTINATION ${ELASTIX_LIBRARY_DIR}
  RUNTIME DESTINATION ${ELASTIX_RUNTIME_DIR} )

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

