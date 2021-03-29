ADD_ELXCOMPONENT(ActiveRegistrationModelShapeMetric OFF
  elxActiveRegistrationModelShapeMetric.cxx
  elxActiveRegistrationModelShapeMetric.h
  elxActiveRegistrationModelShapeMetric.hxx
  itkActiveRegistrationModelShapeMetric.h
  itkActiveRegistrationModelShapeMetric.hxx
)

ADD_ELXCOMPONENT(ActiveRegistrationModelIntensityMetric OFF
  elxActiveRegistrationModelIntensityMetric.h
  elxActiveRegistrationModelIntensityMetric.hxx
  elxActiveRegistrationModelIntensityMetric.cxx
  itkActiveRegistrationModelIntensityMetric.h
  itkActiveRegistrationModelIntensityMetric.hxx
)

# Eigen3, statismo and hdf5 include directories and Eigen3 and hdf5 libraries are transitively included via
# the statismo_core target
if(${USE_ActiveRegistrationModelShapeMetric} OR ${USE_ActiveRegistrationModelIntensityMetric})
  add_subdirectory(Statismo)
endif()

if(USE_ActiveRegistrationModelShapeMetric)
  target_link_libraries(ActiveRegistrationModelShapeMetric statismo_core ${Boost_LIBRARIES} ITKInternalEigen3::Eigen)
endif()

if(USE_ActiveRegistrationModelIntensityMetric)
  target_link_libraries(ActiveRegistrationModelIntensityMetric statismo_core ${Boost_LIBRARIES} ITKInternalEigen3::Eigen)
endif()
