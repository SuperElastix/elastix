PROJECT(elxCommon)

SUBDIRS(ParameterFileParser)
SUBDIRS(xout)


SET(CommonFiles
elxTimer.cxx
elxTimer.h
elx_sample.cxx
elx_sample.h
itkImageNotSoRandomConstIteratorWithIndex.h
itkImageNotSoRandomIteratorWithIndex.h
itkMaskImage.h
itkMaskImage.hxx
itkNDImageBase.h
itkNDImageTemplate.h
itkNDImageTemplate.hxx 
)

SET( TransformGrouperFiles 
TransformGrouper/itkBSplineTransformGrouper.h
TransformGrouper/itkBSplineTransformGrouper.hxx
TransformGrouper/itkTransformGrouper.h
TransformGrouper/itkTransformGrouper.hxx
TransformGrouper/itkTransformGrouperInterface.h )


SOURCE_GROUP("TransformGrouper" FILES ${TransformGrouperFiles} )

ADD_LIBRARY(elxCommon SHARED ${CommonFiles} ${TransformGrouperFiles} )

TARGET_LINK_LIBRARIES(elxCommon
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )
