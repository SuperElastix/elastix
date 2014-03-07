/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkImageToImageMetricWithFeatures_h
#define __itkImageToImageMetricWithFeatures_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkInterpolateImageFunction.h"


namespace itk
{

/** \class ImageToImageMetricWithFeatures
 * \brief Computes similarity between regions of two images.
 *
 * This base class adds functionality that makes it possible
 * to use fixed and moving image features.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage,
  class TFixedFeatureImage = TFixedImage, class TMovingFeatureImage = TMovingImage>
class ImageToImageMetricWithFeatures :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:
  /** Standard class typedefs. */
  typedef ImageToImageMetricWithFeatures  Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >           Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageToImageMetricWithFeatures, AdvancedImageToImageMetric );

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
  typedef typename Superclass::InternalMaskPixelType      InternalMaskPixelType;
  typedef typename
    Superclass::InternalMovingImageMaskType               InternalMovingImageMaskType;
  typedef typename
    Superclass::MovingImageMaskInterpolatorType           MovingImageMaskInterpolatorType;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Typedefs for the feature images. */
  typedef TFixedFeatureImage                                FixedFeatureImageType;
  typedef typename FixedFeatureImageType::Pointer           FixedFeatureImagePointer;
  typedef TMovingFeatureImage                               MovingFeatureImageType;
  typedef typename MovingFeatureImageType::Pointer          MovingFeatureImagePointer;
  typedef std::vector<FixedFeatureImagePointer>             FixedFeatureImageVectorType;
  typedef std::vector<MovingFeatureImagePointer>            MovingFeatureImageVectorType;

  /** Typedefs for the feature images interpolators. */
  typedef InterpolateImageFunction<
    FixedFeatureImageType,double>                           FixedFeatureInterpolatorType;
  typedef InterpolateImageFunction<
    MovingFeatureImageType,double>                          MovingFeatureInterpolatorType;
  typedef typename FixedFeatureInterpolatorType::Pointer    FixedFeatureInterpolatorPointer;
  typedef typename MovingFeatureInterpolatorType::Pointer   MovingFeatureInterpolatorPointer;
  typedef std::vector<FixedFeatureInterpolatorPointer>      FixedFeatureInterpolatorVectorType;
  typedef std::vector<MovingFeatureInterpolatorPointer>     MovingFeatureInterpolatorVectorType;

  /** Set the number of fixed feature images. */
  void SetNumberOfFixedFeatureImages( unsigned int arg );

  /** Get the number of fixed feature images. */
  itkGetConstMacro( NumberOfFixedFeatureImages, unsigned int );

  /** Functions to set the fixed feature images. */
  void SetFixedFeatureImage( unsigned int i, FixedFeatureImageType * im );
  void SetFixedFeatureImage( FixedFeatureImageType * im )
  {
    this->SetFixedFeatureImage( 0, im );
  };

  /** Functions to get the fixed feature images. */
  const FixedFeatureImageType * GetFixedFeatureImage( unsigned int i ) const;
  const FixedFeatureImageType * GetFixedFeatureImage( void ) const
  {
    return this->GetFixedFeatureImage( 0 );
  };

  /** Functions to set the fixed feature interpolators. */
  void SetFixedFeatureInterpolator( unsigned int i, FixedFeatureInterpolatorType * interpolator );
  void SetFixedFeatureInterpolator( FixedFeatureInterpolatorType * interpolator )
  {
    this->SetFixedFeatureInterpolator( 0, interpolator );
  };

  /** Functions to get the fixed feature interpolators. */
  const FixedFeatureInterpolatorType * GetFixedFeatureInterpolator( unsigned int i ) const;
  const FixedFeatureInterpolatorType * GetFixedFeatureInterpolator( void ) const
  {
    return this->GetFixedFeatureInterpolator( 0 );
  };

  /** Set the number of moving feature images. */
  void SetNumberOfMovingFeatureImages( unsigned int arg );

  /** Get the number of moving feature images. */
  itkGetConstMacro( NumberOfMovingFeatureImages, unsigned int );

  /** Functions to set the moving feature images. */
  void SetMovingFeatureImage( unsigned int i, MovingFeatureImageType * im );
  void SetMovingFeatureImage( MovingFeatureImageType * im )
  {
    this->SetMovingFeatureImage( 0, im );
  };

  /** Functions to get the moving feature images. */
  const MovingFeatureImageType * GetMovingFeatureImage( unsigned int i ) const;
  const MovingFeatureImageType * GetMovingFeatureImage( void ) const
  {
    return this->GetMovingFeatureImage( 0 );
  };

  /** Functions to set the moving feature interpolators. */
  void SetMovingFeatureInterpolator( unsigned int i, MovingFeatureInterpolatorType * interpolator );
  void SetMovingFeatureInterpolator( MovingFeatureInterpolatorType * interpolator )
  {
    this->SetMovingFeatureInterpolator( 0, interpolator );
  };

  /** Functions to get the moving feature interpolators. */
  const MovingFeatureInterpolatorType * GetMovingFeatureInterpolator( unsigned int i ) const;
  const MovingFeatureInterpolatorType * GetMovingFeatureInterpolator( void ) const
  {
    return this->GetMovingFeatureInterpolator( 0 );
  };

  /** Initialize the metric. */
  virtual void Initialize( void ) throw ( ExceptionObject );

protected:
  ImageToImageMetricWithFeatures();
  virtual ~ImageToImageMetricWithFeatures() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  typedef typename Superclass::BSplineInterpolatorType    BSplineInterpolatorType;
  typedef typename BSplineInterpolatorType::Pointer       BSplineInterpolatorPointer;
  typedef std::vector<BSplineInterpolatorPointer>         BSplineFeatureInterpolatorVectorType;
  typedef typename Superclass::FixedImagePointType        FixedImagePointType;
  typedef typename Superclass::MovingImagePointType       MovingImagePointType;
  typedef typename Superclass::MovingImageDerivativeType  MovingImageDerivativeType;
  typedef typename Superclass::MovingImageContinuousIndexType  MovingImageContinuousIndexType;

  /** Member variables. */
  unsigned int                          m_NumberOfFixedFeatureImages;
  unsigned int                          m_NumberOfMovingFeatureImages;
  FixedFeatureImageVectorType           m_FixedFeatureImages;
  MovingFeatureImageVectorType          m_MovingFeatureImages;
  FixedFeatureInterpolatorVectorType    m_FixedFeatureInterpolators;
  MovingFeatureInterpolatorVectorType   m_MovingFeatureInterpolators;

  std::vector<bool>                     m_FeatureInterpolatorsIsBSpline;
  bool                                  m_FeatureInterpolatorsAreBSpline;
  BSplineFeatureInterpolatorVectorType  m_MovingFeatureBSplineInterpolators;

  /** Initialize variables for image derivative computation; this
   * method is called by Initialize.
   */
  virtual void CheckForBSplineFeatureInterpolators( void );

private:
  ImageToImageMetricWithFeatures(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class ImageToImageMetricWithFeatures

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToImageMetricWithFeatures.txx"
#endif

#endif // end #ifndef __itkImageToImageMetricWithFeatures_h



