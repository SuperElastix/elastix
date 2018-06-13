/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxAdvancedNormalizedCorrelationMetric_H__
#define __elxAdvancedNormalizedCorrelationMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedNormalizedCorrelationImageToImageMetric.h"

#include "elxTimer.h"

namespace elastix
{

/**
 * \class AdvancedNormalizedCorrelationMetric
 * \brief An metric based on the itk::AdvancedNormalizedCorrelationImageToImageMetric.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "AdvancedNormalizedCorrelation")</tt>
 * \parameter SubtractMean: Flag to set if the sample mean is subtracted from the
 *    sample values in the cross correlation formula. This typically results in narrower
 *    valleys in the cost function. Default value is true. Can be defined for each resolution\n
 *    example: <tt>(SubtractMean "false")</tt>
 *
 * \ingroup Metrics
 *
 */

template< class TElastix >
class AdvancedNormalizedCorrelationMetric :
  public
  itk::AdvancedNormalizedCorrelationImageToImageMetric<
  typename MetricBase< TElastix >::FixedImageType,
  typename MetricBase< TElastix >::MovingImageType >,
  public MetricBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef AdvancedNormalizedCorrelationMetric Self;
  typedef itk::AdvancedNormalizedCorrelationImageToImageMetric<
    typename MetricBase< TElastix >::FixedImageType,
    typename MetricBase< TElastix >::MovingImageType >    Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedNormalizedCorrelationMetric, itk::AdvancedNormalizedCorrelationImageToImageMetric );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "AdvancedNormalizedCorrelation")</tt>\n
   */
  elxClassNameMacro( "AdvancedNormalizedCorrelation" );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::MovingImageType            MovingImageType;
  typedef typename Superclass1::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass1::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass1::FixedImageType             FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType              TransformType;
  typedef typename Superclass1::TransformPointer           TransformPointer;
  typedef typename Superclass1::InputPointType             InputPointType;
  typedef typename Superclass1::OutputPointType            OutputPointType;
  typedef typename Superclass1::TransformParametersType    TransformParametersType;
  typedef typename Superclass1::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass1::InterpolatorType           InterpolatorType;
  typedef typename Superclass1::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::GradientPixelType          GradientPixelType;
  typedef typename Superclass1::GradientImageType          GradientImageType;
  typedef typename Superclass1::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass1::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass1::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass1::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                MeasureType;
  typedef typename Superclass1::DerivativeType             DerivativeType;
  typedef typename Superclass1::ParametersType             ParametersType;
  typedef typename Superclass1::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass1::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass1::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass1::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass1::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass1::ImageSampleContainerPointer ImageSampleContainerPointer;
  typedef typename Superclass1::FixedImageLimiterType  FixedImageLimiterType;
  typedef typename Superclass1::MovingImageLimiterType MovingImageLimiterType;
  typedef typename
    Superclass1::FixedImageLimiterOutputType FixedImageLimiterOutputType;
  typedef typename
    Superclass1::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename
    Superclass1::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Typedef's inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef for timer. */
  typedef tmr::Timer TimerType;
  /** Typedef for timer. */
  typedef TimerType::Pointer TimerPointer;

  /** Execute stuff before each new pyramid resolution:
   * \li Set the flag to subtract the mean.
   * \li Set the CheckNumberOfSamples setting.
   */
  virtual void BeforeEachResolution( void );

  /** Sets up a timer to measure the intialisation time and
   * calls the Superclass' implementation.
   */
  virtual void Initialize( void ) throw ( itk::ExceptionObject );

  /** Switch grid shift strategy. */
  virtual void SetUseGridShiftStrategy( bool useStrategy );

  /** Initialize random shift list. */
  virtual void SetRandomShiftList( std::vector< double > randomList );

protected:

  /** The constructor. */
  AdvancedNormalizedCorrelationMetric() {}
  /** The destructor. */
  virtual ~AdvancedNormalizedCorrelationMetric() {}

private:

  /** The private constructor. */
  AdvancedNormalizedCorrelationMetric( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );               // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAdvancedNormalizedCorrelationMetric.hxx"
#endif

#endif // end #ifndef __elxAdvancedNormalizedCorrelationMetric_H__
