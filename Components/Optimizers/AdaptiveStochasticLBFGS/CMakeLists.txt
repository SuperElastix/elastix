
ADD_ELXCOMPONENT( AdaptiveStochasticLBFGS OFF # OFF by default after elastix 5.0.1
 elxAdaptiveStochasticLBFGS.h
 elxAdaptiveStochasticLBFGS.hxx
 elxAdaptiveStochasticLBFGS.cxx
 itkAdaptiveStochasticLBFGSOptimizer.h
 itkAdaptiveStochasticLBFGSOptimizer.cxx
 ../StandardStochasticGradientDescent/itkStandardStochasticGradientDescentOptimizer.h
 ../StandardStochasticGradientDescent/itkStandardStochasticGradientDescentOptimizer.cxx
 ../StandardStochasticGradientDescent/itkStochasticGradientDescentOptimizer.h
 ../StandardStochasticGradientDescent/itkStochasticGradientDescentOptimizer.cxx
)

if(USE_AdaptiveStochasticLBFGS AND ELASTIX_USE_EIGEN)
  target_include_directories(AdaptiveStochasticLBFGS PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_compile_definitions(AdaptiveStochasticLBFGS PRIVATE ELASTIX_USE_EIGEN)
endif()
