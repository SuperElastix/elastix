PROJECT(elxCommon)

SUBDIRS(ParameterFileParser)
SUBDIRS(xout)


SET(CommonFiles
elxTimer.cxx
elxTimer.h
itkMaskImage.h
itkMaskImage.hxx
itkNDImageBase.h
itkNDImageTemplate.h
itkNDImageTemplate.hxx
itkDeformationFieldRegulizer.h
itkDeformationFieldRegulizer.hxx
itkDeformationFieldRegulizerForBSpline.h
itkDeformationFieldRegulizerForBSpline.hxx
itkDeformationVectorFieldTransform.h
itkDeformationVectorFieldTransform.txx )

SET( TransformGrouperFiles 
TransformGrouper/itkBSplineTransformGrouper.h
TransformGrouper/itkBSplineTransformGrouper.hxx
TransformGrouper/itkTransformGrouper.h
TransformGrouper/itkTransformGrouper.hxx
TransformGrouper/itkTransformGrouperInterface.h )

SET( MoreRandomFiles
MoreRandom/elx_sample.h
MoreRandom/elx_sample.cxx
MoreRandom/vnl_random.h
MoreRandom/vnl_random.cxx
MoreRandom/itkImageMoreRandomConstIteratorWithIndex.h
MoreRandom/itkImageMoreRandomConstIteratorWithIndex.txx
MoreRandom/itkImageMoreRandomIteratorWithIndex.h
MoreRandom/itkImageMoreRandomIteratorWithIndex.txx)

SOURCE_GROUP("TransformGrouper" FILES ${TransformGrouperFiles} )
SOURCE_GROUP("MoreRandom" FILES ${MoreRandomFiles} )

ADD_LIBRARY(elxCommon ${CommonFiles} ${TransformGrouperFiles} ${MoreRandomFiles})

TARGET_LINK_LIBRARIES(elxCommon
ITKBasicFilters
ITKNumerics
ITKIO
ITKCommon )
