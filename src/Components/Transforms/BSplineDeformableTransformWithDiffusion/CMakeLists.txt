#---------------------------------------------------------------------
# Needed for registering this component to elastix
# See src/Components/CMakeList.txt for the macro

ADD_ELXCOMPONENT( elxBSplineTransformWithDiffusion
 elxBSplineTransformWithDiffusion.h
 elxBSplineTransformWithDiffusion.hxx
 elxBSplineTransformWithDiffusion.cxx
 itkVectorMeanDiffusionImageFilter.h
 itkVectorMeanDiffusionImageFilter.hxx)
