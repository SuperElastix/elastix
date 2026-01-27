ADD_ELXCOMPONENT(ImpactMetric OFF PLUGIN
 elxImpactMetric.cxx
 elxImpactMetric.h
 elxImpactMetric.hxx
 ImpactLoss.h
 ImpactTensorUtils.h
 ImpactTensorUtils.hxx
 itkBSplineInterpolateVectorImageFunction.h
 itkBSplineInterpolateVectorImageFunction.hxx
 itkImpactImageToImageMetric.h
 itkImpactImageToImageMetric.hxx
 itkImpactModelConfiguration.h
 )

if(USE_ImpactMetric)
    find_package(Torch REQUIRED)
    target_link_libraries(ImpactMetric ${TORCH_LIBRARIES})
endif()
