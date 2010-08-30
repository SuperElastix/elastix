/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkAdvancedTransformRigidityPenaltyTerm_h
#define __itkAdvancedTransformRigidityPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/**
 * \class AdvancedTransformRigidityPenaltyTerm
 * \brief A cost function that calculates the rigidity
 * of a transformation.
 *
 * \par The rigidity is defined as the sum of the spatial
 * second order derivatives of the transformation, as defined in
 * [1]. For rigid and affine transformation this energy is always
 * zero.
 *
 *
 * [1]: ?
 *
 * \ingroup Metrics
 */

template< class TFixedImage, class TScalarType >
class AdvancedTransformRigidityPenaltyTerm
  : public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard ITK stuff. */
  typedef AdvancedTransformRigidityPenaltyTerm  Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedTransformRigidityPenaltyTerm, TransformPenaltyTerm );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImagePointer          FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
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
  typedef typename Superclass::ScalarType                 ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Get the penalty term value. */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** Get the penalty term derivative. */
  virtual void GetDerivative( const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get the penalty term value and derivative. */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  /** Set/Get the weight of the linearity condition part. */
  itkSetClampMacro( LinearityConditionWeight, ScalarType,
    NumericTraits<ScalarType>::Zero, NumericTraits<ScalarType>::max() );
  itkGetMacro( LinearityConditionWeight, ScalarType );

  /** Set/Get the weight of the orthonormality condition part. */
  itkSetClampMacro( OrthonormalityConditionWeight, ScalarType,
    NumericTraits<ScalarType>::Zero, NumericTraits<ScalarType>::max() );
  itkGetMacro( OrthonormalityConditionWeight, ScalarType );

  /** Set/Get the weight of the properness condition part. */
  itkSetClampMacro( PropernessConditionWeight, ScalarType,
    NumericTraits<ScalarType>::Zero, NumericTraits<ScalarType>::max() );
  itkGetMacro( PropernessConditionWeight, ScalarType );

  /** Set the usage of the linearity condition part. */
  itkSetMacro( UseLinearityCondition, bool );

  /** Set the usage of the orthonormality condition part. */
  itkSetMacro( UseOrthonormalityCondition, bool );

  /** Set the usage of the properness condition part. */
  itkSetMacro( UsePropernessCondition, bool );

  /** Set the calculation of the linearity condition part,
   * even if we don't use it.
   */
  itkSetMacro( CalculateLinearityCondition, bool );

  /** Set the calculation of the orthonormality condition part,
   * even if we don't use it.
   */
  itkSetMacro( CalculateOrthonormalityCondition, bool );

  /** Set the calculation of the properness condition part.,
   * even if we don't use it.
   */
  itkSetMacro( CalculatePropernessCondition, bool );

  /** Get the value of the linearity condition. */
  itkGetConstReferenceMacro( LinearityConditionValue, MeasureType );

  /** Get the value of the orthonormality condition. */
  itkGetConstReferenceMacro( OrthonormalityConditionValue, MeasureType );

  /** Get the value of the properness condition. */
  itkGetConstReferenceMacro( PropernessConditionValue, MeasureType );

protected:

  /** Typedefs for indices and points. */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;

  /** The constructor. */
  AdvancedTransformRigidityPenaltyTerm();

  /** The destructor. */
  virtual ~AdvancedTransformRigidityPenaltyTerm() {};

  /** Typedef's for the B-spline transform. */
  typedef typename Superclass::BSplineTransformType       BSplineTransformType;
  typedef typename Superclass::CombinationTransformType   CombinationTransformType;

private:

  /** The private constructor. */
  AdvancedTransformRigidityPenaltyTerm( const Self& ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );                    // purposely not implemented

  /** Member variables for the several parts of the penalty term. */
  ScalarType              m_LinearityConditionWeight;
  ScalarType              m_OrthonormalityConditionWeight;
  ScalarType              m_PropernessConditionWeight;
  bool                    m_UseLinearityCondition;
  bool                    m_UseOrthonormalityCondition;
  bool                    m_UsePropernessCondition;
  bool                    m_CalculateLinearityCondition;
  bool                    m_CalculateOrthonormalityCondition;
  bool                    m_CalculatePropernessCondition;

  mutable MeasureType     m_LinearityConditionValue;
  mutable MeasureType     m_OrthonormalityConditionValue;
  mutable MeasureType     m_PropernessConditionValue;

}; // end class AdvancedTransformRigidityPenaltyTerm


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedTransformRigidityPenaltyTerm.txx"
#endif

#endif // #ifndef __itkAdvancedTransformRigidityPenaltyTerm_h

