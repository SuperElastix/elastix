/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/


#ifndef __itkZeroDeformationConstraintMetric_h
#define __itkZeroDeformationConstraintMetric_h

#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkTransformPenaltyTerm.h"

namespace itk
{

/** \class ZeroDeformationConstraintMetric
 * \brief Penalizes deformation using an augmented Lagrangian penalty term.
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template < class TFixedImage, class TMovingImage >
class ZeroDeformationConstraintMetric :
    public TransformPenaltyTerm< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef ZeroDeformationConstraintMetric            Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TMovingImage >                   Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( VarianceOverLastDimensionImageMetric, AdvancedImageToImageMetric );

  /** Set functions. */
  itkSetMacro( CurrentPenaltyTermMultiplier, float );
  itkSetMacro( InitialLangrangeMultiplier, float );

  /** Get functions. */
  itkGetConstMacro( CurrentPenaltyTermMultiplier, float );
  itkGetConstMacro( CurrentMaximumAbsoluteDisplacement, float );
  itkGetConstMacro( InitialLangrangeMultiplier, float );
  itkGetConstMacro( CurrentInfeasibility, float );

  float GetCurrentPenaltyTermValue ( const int i ) const {
    return m_CurrentPenaltyTermValues[ i ];
  }

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
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
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
  typedef typename
    Superclass::NonZeroJacobianIndicesType                NonZeroJacobianIndicesType;
  typedef typename
    Superclass::MovingImageDerivativeScalesType           MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Get the value for single valued optimizers. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  virtual void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation.   */
  virtual void Initialize(void) throw ( ExceptionObject );

protected:
  ZeroDeformationConstraintMetric();
  virtual ~ZeroDeformationConstraintMetric() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
  typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension >
                                                                  FixedImageContinuousIndexType;
  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType            BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType          MovingImageDerivativeType;

  /** Current penalty term value. */
  mutable std::vector< float > m_CurrentPenaltyTermValues;

  /** Current lagrange multipliers (per constraint) and penalty term multiplier. */
  float m_CurrentPenaltyTermMultiplier;
  std::vector< float > m_CurrentLagrangeMultipliers;

  /** Current infeasibility value: sum( abs ( c( x_k ) ) ) / K. */
  mutable float m_CurrentInfeasibility;

  /** Current maximum absolute displacement value. */
  mutable float m_CurrentMaximumAbsoluteDisplacement;

  /** Initial langrange multiplier value. */
  float m_InitialLangrangeMultiplier;

private:
  ZeroDeformationConstraintMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class ZeroDeformationConstraintMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkZeroDeformationConstraint.hxx"
#endif

#endif // end #ifndef __itkZeroDeformationConstraintMetric_h
