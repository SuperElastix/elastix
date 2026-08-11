ADD_ELXCOMPONENT(ImpactMetric OFF PLUGIN
 elxImpactMetric.cxx
 elxImpactMetric.h
 elxImpactMetric.hxx
 itkImpactImageToImageMetric.h
 itkImpactImageToImageMetric.hxx
 )

if(USE_ImpactMetric)
    find_package(Torch REQUIRED)
    # IMPACT losses, model configuration and online inference come from the ITKIMPACT
    # remote module (one-way dependency Elastix -> ITKIMPACT).
    set(ITKIMPACT_INCLUDE_DIR "" CACHE PATH
        "ITKIMPACT backend include directory (provides ImpactLoss.h, itkModelConfiguration.h, ...)")
    # Check it before use: an unset ITKIMPACT_INCLUDE_DIR collapses the path below to
    # /src/itkModelConfiguration.cxx, and CMake then reports a missing source file rather than
    # the missing setting that caused it.
    if(NOT EXISTS "${ITKIMPACT_INCLUDE_DIR}/ImpactLoss.h")
        message(FATAL_ERROR
            "USE_ImpactMetric is ON, so the ITKIMPACT backend is required, but "
            "ITKIMPACT_INCLUDE_DIR does not point at it: '${ITKIMPACT_INCLUDE_DIR}'. "
            "Clone https://github.com/InsightSoftwareConsortium/ITKIMPACT and set "
            "ITKIMPACT_INCLUDE_DIR to its include/ directory.")
    endif()
    target_include_directories(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR})
    target_sources(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR}/../src/itkModelConfiguration.cxx)
    target_link_libraries(ImpactMetric PUBLIC ${TORCH_LIBRARIES})
endif()
