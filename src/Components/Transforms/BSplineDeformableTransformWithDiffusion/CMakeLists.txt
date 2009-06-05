#---------------------------------------------------------------------
# Needed for registering this component to elastix
# See src/Components/CMakeList.txt for the macro

ADD_ELXCOMPONENT( BSplineTransformWithDiffusion
 elxBSplineTransformWithDiffusion.h
 elxBSplineTransformWithDiffusion.hxx
 elxBSplineTransformWithDiffusion.cxx
 itkDeformationFieldRegulizer.h
 itkDeformationFieldRegulizer.hxx
 itkDeformationFieldRegulizerForBSpline.h
 itkDeformationFieldRegulizerForBSpline.hxx
 itkDeformationVectorFieldTransform.h
 itkDeformationVectorFieldTransform.txx
 itkVectorMeanDiffusionImageFilter.h
 itkVectorMeanDiffusionImageFilter.hxx)
