
# Only compile this component when the SuiteSparse library is found
if( SuiteSparse_FOUND )
  ADD_ELXCOMPONENT( PreconditionedGradientDescent OFF
    elxPreconditionedGradientDescent.h
    elxPreconditionedGradientDescent.hxx
    elxPreconditionedGradientDescent.cxx
    itkAdaptiveStochasticPreconditionedGradientDescentOptimizer.h
    itkAdaptiveStochasticPreconditionedGradientDescentOptimizer.cxx
    itkStochasticPreconditionedGradientDescentOptimizer.h
    itkStochasticPreconditionedGradientDescentOptimizer.cxx
    itkPreconditionedGradientDescentOptimizer.h
    itkPreconditionedGradientDescentOptimizer.cxx )

  if( USE_PreconditionedGradientDescent )
    target_link_libraries( PreconditionedGradientDescent ${SuiteSparse_LIBRARIES} )
  endif()
else()
  # If the user set USE_PreconditionedGradientDescent ON,
  # but SuiteSparse_FOUND was FALSE, then issue a warning.
  if( USE_PreconditionedGradientDescent )
    message( WARNING "You selected to compile PreconditionedGradientDescent, "
      "but SuiteSparse_FOUND is OFF.\n"
      "Set both options to ON to be able to build this component." )
  endif()

  # If SuiteSparse_FOUND is FALSE, then we cannot compile this component.
  set( USE_PreconditionedGradientDescent OFF CACHE BOOL "Compile this component" FORCE )
  mark_as_advanced( USE_PreconditionedGradientDescent )

  # This is required to get the PreconditionedGradientDescent out of the
  # AllComponentLibs list defined in Components/CMakeLists.txt.
  REMOVE_ELXCOMPONENT( USE_PreconditionedGradientDescent )
endif()

