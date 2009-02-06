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

template< class TFixedImage, class TScalarType >
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
  typedef typename Superclass::MeasureType          MeasureType;
  typedef typename Superclass::RealType             RealType;
  typedef typename Superclass::DerivativeType       DerivativeType;
  typedef typename Superclass::ParametersType       ParametersType;
  typedef typename Superclass::ImageSampleContainerType    ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer ImageSampleContainerPointer;

  /** Template parameters. */
  typedef TFixedImage   FixedImageType;
  typedef TScalarType		ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass
    ::AdvancedTransformType                     TransformType;
  typedef typename TransformType
    ::NonZeroJacobianIndicesType                NonZeroJacobianIndicesType;
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

  /** Initialize the penalty term by making sure that
   * all the components are present and plugged together correctly.
   */
  virtual void Initialize( void ) throw ( ExceptionObject );

protected:

  /** The constructor. */
  TransformPenaltyTerm();

  /** The destructor. */
  virtual ~TransformPenaltyTerm() {};

  /** PrintSelf. */
  void PrintSelf( std::ostream& os, Indent indent ) const;

  typename TransformType::Pointer m_AdvancedTransform;

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

