/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkPatternIntensityImageToImageMetric_h
#define __itkPatternIntensityImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

#include "itkPoint.h"
#include "itkCastImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkOptimizer.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"

namespace itk
{

/** \class PatternIntensityImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 *
 *
 * \ingroup RegistrationMetrics
 */

template< class TFixedImage, class TMovingImage >
class PatternIntensityImageToImageMetric :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef PatternIntensityImageToImageMetric Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                 Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( PatternIntensityImageToImageMetric, ImageToImageMetric );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename TransformType::ScalarType              ScalarType;
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
    Superclass::ImageSampleContainerPointer ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType  FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;
  typedef typename itk::Optimizer            OptimizerType;
  typedef typename OptimizerType::ScalesType ScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  typedef itk::Image< FixedImagePixelType,
    itkGetStaticConstMacro( FixedImageDimension ) >       TransformedMovingImageType;
  typedef typename itk::AdvancedCombinationTransform<
    ScalarType, FixedImageDimension >                      CombinationTransformType;
  typedef typename CombinationTransformType::Pointer CombinationTransformPointer;
  typedef typename itk::AdvancedRayCastInterpolateImageFunction<
    MovingImageType, ScalarType >                         RayCastInterpolatorType;
  typedef typename RayCastInterpolatorType::Pointer RayCastInterpolatorPointer;
  typedef itk::ResampleImageFilter<
    MovingImageType, TransformedMovingImageType >         TransformMovingImageFilterType;
  typedef typename TransformMovingImageFilterType::Pointer TransformMovingImageFilterPointer;
  typedef itk::RescaleIntensityImageFilter<
    TransformedMovingImageType, TransformedMovingImageType > RescaleIntensityImageFilterType;
  typedef typename RescaleIntensityImageFilterType::Pointer RescaleIntensityImageFilterPointer;

  typedef itk::SubtractImageFilter<
    FixedImageType,
    TransformedMovingImageType,
    TransformedMovingImageType >        DifferenceImageFilterType;
  typedef typename DifferenceImageFilterType::Pointer DifferenceImageFilterPointer;
  typedef itk::MultiplyImageFilter<
    TransformedMovingImageType,
    TransformedMovingImageType,
    TransformedMovingImageType >         MultiplyImageFilterType;
  typedef typename MultiplyImageFilterType::Pointer MultiplyImageFilterPointer;

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
    MeasureType & Value, DerivativeType & Derivative ) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation
   * \li Estimate the normalization factor, if asked for.
   */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Set/Get Scales  */
  itkSetMacro( Scales, ScalesType );
  itkGetConstReferenceMacro( Scales, ScalesType );

  /** Set/Get m_NoiseConstant  */
  itkSetMacro( NoiseConstant, double );
  itkGetConstReferenceMacro( NoiseConstant, double );

  /** Set/Get OptimizeNormalizationFactor  */
  itkSetMacro( OptimizeNormalizationFactor, bool );
  itkGetConstReferenceMacro( OptimizeNormalizationFactor, bool );

protected:

  PatternIntensityImageToImageMetric();
  virtual ~PatternIntensityImageToImageMetric() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Compute the pattern intensity fixed image*/
  MeasureType ComputePIFixed( void ) const;

  /** Compute the pattern intensity difference image. */
  MeasureType ComputePIDiff( const TransformParametersType & parameters, float scalingfactor ) const;

private:

  PatternIntensityImageToImageMetric( const Self & ); // purposely not implemented
  void operator=( const Self & );                     // purposely not implemented

  TransformMovingImageFilterPointer  m_TransformMovingImageFilter;
  DifferenceImageFilterPointer       m_DifferenceImageFilter;
  RescaleIntensityImageFilterPointer m_RescaleImageFilter;
  MultiplyImageFilterPointer         m_MultiplyImageFilter;
  double                             m_NoiseConstant;
  unsigned int                       m_NeighborhoodRadius;
  double                             m_DerivativeDelta;
  double                             m_NormalizationFactor;
  double                             m_Rescalingfactor;
  bool                               m_OptimizeNormalizationFactor;
  ScalesType                         m_Scales;
  MeasureType                        m_FixedMeasure;
  CombinationTransformPointer        m_CombinationTransform;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPatternIntensityImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkPatternIntensityImageToImageMetric_h
