/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkTransformPenaltyTerm_h
#define __itkTransformPenaltyTerm_h

#include "itkAdvancedImageToImageMetric.h"

// Needed for checking for B-spline for faster implementation
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"


namespace itk
{
/**
 * \class TransformPenaltyTerm
 * \brief A cost function that calculates a penalty term
 * on a transformation.
 *
 * \par We decided to make it an itk::ImageToImageMetric, since possibly
 * all the stuff in there is also needed for penalty terms.
 *
 * A transformation penalty terms has some extra demands on the transform.
 * Therefore, the transformation is required to be of itk::AdvancedTransform
 * type.
 *
 * \ingroup Metrics
 */

template< class TFixedImage, class TScalarType = double >
class TransformPenaltyTerm
  : public AdvancedImageToImageMetric< TFixedImage, TFixedImage >
{
public:

  /** Standard ITK stuff. */
  typedef TransformPenaltyTerm				    Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TFixedImage >            Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformPenaltyTerm, AdvancedImageToImageMetric );

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImagePointer          FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  // these not: use advanced transform below
  //typedef typename Superclass::TransformType              TransformType;
  //typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType    ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer ImageSampleContainerPointer;

  /** Template parameters. FixedImageType has already been taken from superclass. */
  typedef TScalarType		ScalarType; // \todo: not really meaningful name.

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass
    ::AdvancedTransformType                     TransformType;
  typedef typename TransformType
    ::SpatialJacobianType                       SpatialJacobianType;
  typedef typename TransformType
    ::JacobianOfSpatialJacobianType             JacobianOfSpatialJacobianType;
  typedef typename TransformType
    ::SpatialHessianType                        SpatialHessianType;
  typedef typename TransformType
    ::JacobianOfSpatialHessianType              JacobianOfSpatialHessianType;
  typedef typename TransformType
    ::InternalMatrixType                        InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

protected:

  /** Typedefs for indices and points. */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  typedef typename Superclass::NonZeroJacobianIndicesType         NonZeroJacobianIndicesType;

  /** The constructor. */
  TransformPenaltyTerm(){};

  /** The destructor. */
  virtual ~TransformPenaltyTerm() {};

  /** PrintSelf. */
  //void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Typedef's for the B-spline transform. */
  typedef AdvancedBSplineDeformableTransform<
    ScalarType, FixedImageDimension, 3 >          BSplineTransformType;
  typedef typename BSplineTransformType::Pointer  BSplineTransformPointer;
  typedef AdvancedCombinationTransform<
    ScalarType, FixedImageDimension >             CombinationTransformType;

  /** A function to check if the transform is B-spline, for speedup. */
  virtual bool CheckForBSplineTransform( BSplineTransformPointer & bspline ) const;

private:

  /** The private constructor. */
  TransformPenaltyTerm( const Self& );	// purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );			  // purposely not implemented

}; // end class TransformPenaltyTerm


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformPenaltyTerm.txx"
#endif

#endif // #ifndef __itkTransformPenaltyTerm_h

