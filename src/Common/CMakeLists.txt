PROJECT(elxCommon)

SUBDIRS(ParameterFileParser)
SUBDIRS(xout)

SET(CommonFiles
itkCombinationTransform.h
itkCombinationTransform.hxx
itkBSplineCombinationTransform.h
itkBSplineCombinationTransform.hxx
elxTimer.cxx
elxTimer.h
itkNDImageBase.h
itkNDImageTemplate.h
itkNDImageTemplate.hxx
itkDeformationFieldRegulizer.h
itkDeformationFieldRegulizer.hxx
itkDeformationFieldRegulizerForBSpline.h
itkDeformationFieldRegulizerForBSpline.hxx
itkDeformationVectorFieldTransform.h
itkDeformationVectorFieldTransform.txx 
itkScaledSingleValuedCostFunction.h
itkScaledSingleValuedCostFunction.cxx
itkScaledSingleValuedNonLinearOptimizer.h
itkScaledSingleValuedNonLinearOptimizer.cxx
itkTranslationTransformInitializer.h
itkTranslationTransformInitializer.hxx)

SET( LineSearchOptimizersFiles
LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.h
LineSearchOptimizers/itkMoreThuenteLineSearchOptimizer.cxx
LineSearchOptimizers/itkLineSearchOptimizer.h
LineSearchOptimizers/itkLineSearchOptimizer.cxx)

SOURCE_GROUP("LineSearchOptimizers" FILES ${LineSearchOptimizersFiles} )

IF(CMAKE_COMPILER_IS_GNUCXX)
  #Add -fPIC to the gcc-options. 
  #This is needed for compilation on a 64-bit machine 
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

ADD_LIBRARY(elxCommon
  ${ELX_GDCMHACK}
  ${CommonFiles}
	${LineSearchOptimizersFiles} )

TARGET_LINK_LIBRARIES(elxCommon
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )

