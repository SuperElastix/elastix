/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkNormalizedGradientCorrelationImageToImageMetric_h
#define __itkNormalizedGradientCorrelationImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkSobelOperator.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkPoint.h"
#include "itkCastImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkOptimizer.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedRayCastInterpolateImageFunction.h"

namespace itk
{

/**
 * \class NormalizedGradientCorrelationImageToImageMetric
 * \brief An metric based on the itk::NormalizedGradientCorrelationImageToImageMetric.
 *
 *
 * \ingroup Metrics
 *
 */

template< class TFixedImage, class TMovingImage >
class NormalizedGradientCorrelationImageToImageMetric :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef NormalizedGradientCorrelationImageToImageMetric         Self;
  typedef AdvancedImageToImageMetric< TFixedImage, TMovingImage > Superclass;
  typedef SmartPointer< Self >                                    Pointer;
  typedef SmartPointer< const Self >                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( NormalizedGradientCorrelationImageToImageMetric, AdvancedImageToImageMetric );

  /** Types transferred from the base class */
  /** Work around a Visual Studio .NET bug */
  #if defined( _MSC_VER ) && ( _MSC_VER == 1300 )
  typedef double RealType;
  #else
  typedef typename Superclass::RealType RealType;
  #endif

  typedef typename Superclass::TransformType           TransformType;
  typedef typename TransformType::ScalarType           ScalarType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename TransformType::ConstPointer         TransformConstPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;
  typedef typename Superclass::InterpolatorType        InterpolatorType;
  typedef typename InterpolatorType::Pointer           InterpolatorPointer;
  typedef typename Superclass::MeasureType             MeasureType;
  typedef typename Superclass::DerivativeType          DerivativeType;
  typedef typename Superclass::FixedImageType          FixedImageType;
  typedef typename Superclass::FixedImageRegionType    FixedImageRegionType;
  typedef typename Superclass::MovingImageType         MovingImageType;
  typedef typename Superclass::MovingImageRegionType   MovingImageRegionType;
  typedef typename Superclass::FixedImageConstPointer  FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer MovingImageConstPointer;
  typedef typename Superclass::MovingImagePointer      MovingImagePointer;
  typedef typename TFixedImage::PixelType              FixedImagePixelType;
  typedef typename TMovingImage::PixelType             MovedImagePixelType;
  typedef typename itk::Optimizer                      OptimizerType;
  typedef typename OptimizerType::ScalesType           ScalesType;

  itkStaticConstMacro( FixedImageDimension, unsigned int, TFixedImage::ImageDimension );

  /** Types for transforming the moving image */
  typedef typename itk::AdvancedCombinationTransform<
    ScalarType, FixedImageDimension >                    CombinationTransformType;
  typedef typename CombinationTransformType::Pointer CombinationTransformPointer;
  typedef itk::Image< FixedImagePixelType,
    itkGetStaticConstMacro( FixedImageDimension ) >   TransformedMovingImageType;
  typedef itk::Image< unsigned char,
    itkGetStaticConstMacro( FixedImageDimension ) >   MaskImageType;
  typedef typename MaskImageType::Pointer MaskImageTypePointer;
  typedef itk::ResampleImageFilter<
    MovingImageType, TransformedMovingImageType >       TransformMovingImageFilterType;
  typedef typename TransformMovingImageFilterType::Pointer TransformMovingImageFilterPointer;
  typedef typename itk::AdvancedRayCastInterpolateImageFunction
    < MovingImageType, ScalarType >                     RayCastInterpolatorType;
  typedef typename RayCastInterpolatorType::Pointer RayCastInterpolatorPointer;

  /** Sobel filters to compute the gradients of the Fixed Image */
  typedef itk::Image< RealType,
    itkGetStaticConstMacro( FixedImageDimension ) >       FixedGradientImageType;
  typedef itk::CastImageFilter< FixedImageType,
    FixedGradientImageType >                              CastFixedImageFilterType;
  typedef typename CastFixedImageFilterType::Pointer CastFixedImageFilterPointer;
  typedef typename FixedGradientImageType::PixelType FixedGradientPixelType;

  /** Sobel filters to compute the gradients of the Moved Image */
  itkStaticConstMacro( MovedImageDimension, unsigned int, MovingImageType::ImageDimension );
  typedef itk::Image< RealType,
    itkGetStaticConstMacro( MovedImageDimension ) >       MovedGradientImageType;
  typedef itk::CastImageFilter< TransformedMovingImageType,
    MovedGradientImageType >                              CastMovedImageFilterType;
  typedef typename CastMovedImageFilterType::Pointer CastMovedImageFilterPointer;
  typedef typename MovedGradientImageType::PixelType MovedGradientPixelType;

  /** Get the derivatives of the match measure. */
  virtual void GetDerivative( const TransformParametersType & parameters,
    DerivativeType  & derivative ) const;

  /**  Get the value for single valued optimizers. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType & Value, DerivativeType & derivative ) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Write gradient images to a files for debugging purposes. */
  void WriteGradientImagesToFiles( void ) const;

  /** Set/Get Scales  */
  itkSetMacro( Scales, ScalesType );
  itkGetConstReferenceMacro( Scales, ScalesType );

  /** Set/Get the value of Delta used for computing derivatives by finite
   * differences in the GetDerivative() method.
   */
  itkSetMacro( DerivativeDelta, double );
  itkGetConstReferenceMacro( DerivativeDelta, double );

  /** Set the parameters defining the Transform. */
  void SetTransformParameters( const TransformParametersType & parameters ) const;

protected:

  NormalizedGradientCorrelationImageToImageMetric();
  virtual ~NormalizedGradientCorrelationImageToImageMetric() {}
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Compute the mean of the fixed and moved image gradients. */
  void ComputeMeanMovedGradient( void ) const;

  void ComputeMeanFixedGradient( void ) const;

  /** Compute the similarity measure  */
  MeasureType ComputeMeasure( const TransformParametersType & parameters ) const;

  typedef NeighborhoodOperatorImageFilter<
    FixedGradientImageType, FixedGradientImageType >        FixedSobelFilter;
  typedef NeighborhoodOperatorImageFilter<
    MovedGradientImageType, MovedGradientImageType >        MovedSobelFilter;

private:

  NormalizedGradientCorrelationImageToImageMetric( const Self & ); // purposely not implemented
  void operator=( const Self & );                                  // purposely not implemented

  ScalesType                  m_Scales;
  double                      m_DerivativeDelta;
  CombinationTransformPointer m_CombinationTransform;

  /** The mean of the moving image gradients. */
  mutable MovedGradientPixelType m_MeanMovedGradient[ MovedImageDimension ];

  /** The mean of the fixed image gradients. */
  mutable FixedGradientPixelType m_MeanFixedGradient[ FixedImageDimension ];

  /** The filter for transforming the moving images. */
  TransformMovingImageFilterPointer m_TransformMovingImageFilter;

  /** The Sobel gradients of the fixed image */
  CastFixedImageFilterPointer m_CastFixedImageFilter;

  SobelOperator< FixedGradientPixelType,
  itkGetStaticConstMacro( FixedImageDimension ) >
  m_FixedSobelOperators[ FixedImageDimension ];

  typename FixedSobelFilter::Pointer m_FixedSobelFilters
  [ itkGetStaticConstMacro( FixedImageDimension ) ];

  ZeroFluxNeumannBoundaryCondition< MovedGradientImageType > m_MovedBoundCond;
  ZeroFluxNeumannBoundaryCondition< FixedGradientImageType > m_FixedBoundCond;

  /** The Sobel gradients of the moving image */
  CastMovedImageFilterPointer m_CastMovedImageFilter;
  SobelOperator< MovedGradientPixelType,
  itkGetStaticConstMacro( MovedImageDimension ) >
  m_MovedSobelOperators[ MovedImageDimension ];

  typename MovedSobelFilter::Pointer m_MovedSobelFilters[
    itkGetStaticConstMacro( MovedImageDimension ) ];

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNormalizedGradientCorrelationImageToImageMetric.hxx"
#endif

#endif
