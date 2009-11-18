/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkCombinedImageToImageMetric_h
#define __itkCombinedImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

namespace itk
{
  
/** \class CombinedImageToImageMetric
 * \brief Combines multiple metrics.
 *
 *
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class CombinedImageToImageMetric :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:
  /** Standard class typedefs. */
  typedef CombinedImageToImageMetric      Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >           Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( CombinedImageToImageMetric, AdvancedImageToImageMetric );

  /** Constants for the image dimensions */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    TMovingImage::ImageDimension );
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::AdvancedTransformType      TransformType;
  typedef typename Superclass::TransformType::Pointer     TransformPointer;
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

  /** Advanced typedefs. */
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  
  /** Typedefs for the metrics. */
  typedef ImageToImageMetric< FixedImageType,
    MovingImageType >                                     MetricType;
  typedef typename MetricType::Pointer                    MetricPointer;
  typedef Superclass                                      AdvancedMetricType;
  typedef typename AdvancedMetricType::Pointer            AdvancedMetricPointer;
  typedef SingleValuedCostFunction                        SingleValuedCostFunctionType;
  typedef typename SingleValuedCostFunctionType::Pointer  SingleValuedCostFunctionPointer;

  /**
   * Get and set the metrics and their weights.
   */

  /** Set the number of metrics to combine. */
  void SetNumberOfMetrics( unsigned int count );

  /** Get the number of metrics to combine. */
  itkGetConstReferenceMacro( NumberOfMetrics, unsigned int );

  /** Set metric i. */
  void SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos );

  /** Get metric i. */
  SingleValuedCostFunctionType * GetMetric( unsigned int count );

  /** Set the weight for metric i. */
  void SetMetricWeight( double weight, unsigned int pos );

  /** Get the weight for metric i. */
  double GetMetricWeight( unsigned int pos );

  /** Get the weight for metric i. */
  MeasureType GetMetricValue( unsigned int pos );

  /**
   * Pass everything to all sub metrics.
   * ImageToImageMetric functions:
   */

  /** Pass the transform to all sub metrics. */
  virtual void SetTransform( TransformType *_arg );

  /** Pass the interpolator to all sub metrics. */
  virtual void SetInterpolator( InterpolatorType *_arg );

  /** Pass the fixed image to all sub metrics. */
  virtual void SetFixedImage( const FixedImageType *_arg );

  /** Pass the fixed image mask to all sub metrics. */
  virtual void SetFixedImageMask( FixedImageMaskType *_arg );

  /** Pass the fixed image region to all sub metrics. */
  virtual void SetFixedImageRegion( const FixedImageRegionType _arg );

  /** Pass the moving image to all sub metrics. */
  virtual void SetMovingImage( const MovingImageType *_arg );

  /** Pass the moving image mask to all sub metrics. */
  virtual void SetMovingImageMask( MovingImageMaskType *_arg );
  
  /** Pass computation of the gradient to all sub metrics. */
  virtual void SetComputeGradient( const bool _arg );

  /** Pass computation of the gradient to all sub metrics. */
  virtual void SetComputeGradientOn( void ){ this->SetComputeGradient( true ); };
  virtual void SetComputeGradientOff( void ){ this->SetComputeGradient( false ); };

  /** Pass initialisation to all sub metrics. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Get the number of pixels considered in the computation. */
  //itkGetConstReferenceMacro( NumberOfPixelsCounted, unsigned long );
  //virtual void SetReferenceCount( int );
  //void SetMetaDataDictionary( const MetaDataDictionary &rhs );

  /**
   * Pass everything to all sub metrics.
   * AdvancedImageToImageMetric functions:
   */

  /** Pass the image sampler to all sub metrics. */
  virtual void SetImageSampler( ImageSamplerType * _arg );

  /** If one of the sub metrics needs a sampler, return true. */
  virtual bool GetUseImageSampler( void ) const;
  
  /** Pass this ratio to all sub metrics. */
  virtual void SetRequiredRatioOfValidSamples( const double _arg );

  /** Pass the fixed image limiter to all sub metrics. */
  virtual void SetFixedImageLimiter( FixedImageLimiterType * _arg );

  /** Pass the moving image limiter to all sub metrics. */
  virtual void SetMovingImageLimiter( MovingImageLimiterType * _arg );

  /** Pass the fixed image limiter range ration to all sub metrics. */
  virtual void SetFixedLimitRangeRatio( const double _arg );

  /** Pass the moving image limiter range ration to all sub metrics. */
  virtual void SetMovingLimitRangeRatio( const double _arg );

  /**
   * Combine all sub metrics by adding them.
   */

  /** The GetValue()-method. */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** The GetDerivative()-method. */
  virtual void GetDerivative(
    const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** The GetValueAndDerivative()-method. */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

protected:
  CombinedImageToImageMetric();
  virtual ~CombinedImageToImageMetric() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Store the metrics and the corresponding weights. */
  unsigned int                                      m_NumberOfMetrics;
  std::vector< SingleValuedCostFunctionPointer >    m_Metrics;
  std::vector< double >                             m_MetricWeights;
  mutable std::vector< MeasureType >                m_MetricValues;
    
private:
  CombinedImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
 
}; // end class CombinedImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCombinedImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkCombinedImageToImageMetric_h



