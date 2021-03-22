/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef elxPatternIntensityMetric_h
#define elxPatternIntensityMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkPatternIntensityImageToImageMetric.h"

namespace elastix
{

/**
 * \class PatternIntensityMetric
 * \brief An metric based on the itk::PatternIntensityImageToImageMetric.
 *
 *
 * \ingroup Metrics
 *
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT PatternIntensityMetric
  : public itk::PatternIntensityImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                   typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef PatternIntensityMetric Self;
  typedef itk::PatternIntensityImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                  typename MetricBase<TElastix>::MovingImageType>
                                        Superclass1;
  typedef MetricBase<TElastix>          Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PatternIntensityMetric, itk::PatternIntensityImageToImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "PatternIntensity")</tt>\n
   */
  elxClassNameMacro("PatternIntensity");

  /** Typedefs from the superclass. */
  typedef typename Superclass1::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass1::MovingImageType                 MovingImageType;
  typedef typename Superclass1::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass1::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass1::FixedImageType                  FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer          FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType            FixedImageRegionType;
  typedef typename Superclass1::TransformType                   TransformType;
  typedef typename Superclass1::TransformPointer                TransformPointer;
  typedef typename Superclass1::InputPointType                  InputPointType;
  typedef typename Superclass1::OutputPointType                 OutputPointType;
  typedef typename Superclass1::TransformParametersType         TransformParametersType;
  typedef typename Superclass1::TransformJacobianType           TransformJacobianType;
  typedef typename Superclass1::InterpolatorType                InterpolatorType;
  typedef typename Superclass1::InterpolatorPointer             InterpolatorPointer;
  typedef typename Superclass1::RealType                        RealType;
  typedef typename Superclass1::GradientPixelType               GradientPixelType;
  typedef typename Superclass1::GradientImageType               GradientImageType;
  typedef typename Superclass1::GradientImagePointer            GradientImagePointer;
  typedef typename Superclass1::GradientImageFilterType         GradientImageFilterType;
  typedef typename Superclass1::GradientImageFilterPointer      GradientImageFilterPointer;
  typedef typename Superclass1::FixedImageMaskType              FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer           FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType             MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer          MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                     MeasureType;
  typedef typename Superclass1::DerivativeType                  DerivativeType;
  typedef typename Superclass1::ParametersType                  ParametersType;
  typedef typename Superclass1::FixedImagePixelType             FixedImagePixelType;
  typedef typename Superclass1::MovingImageRegionType           MovingImageRegionType;
  typedef typename Superclass1::ImageSamplerType                ImageSamplerType;
  typedef typename Superclass1::ImageSamplerPointer             ImageSamplerPointer;
  typedef typename Superclass1::ImageSampleContainerType        ImageSampleContainerType;
  typedef typename Superclass1::ImageSampleContainerPointer     ImageSampleContainerPointer;
  typedef typename Superclass1::FixedImageLimiterType           FixedImageLimiterType;
  typedef typename Superclass1::MovingImageLimiterType          MovingImageLimiterType;
  typedef typename Superclass1::FixedImageLimiterOutputType     FixedImageLimiterOutputType;
  typedef typename Superclass1::MovingImageLimiterOutputType    MovingImageLimiterOutputType;
  typedef typename Superclass1::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize(void) override;

  /**
   * Do some things before each resolution:
   * \li Set CheckNumberOfSamples setting
   * \li Set UseNormalization setting
   */
  void
  BeforeRegistration(void) override;

  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  PatternIntensityMetric() = default;
  /** The destructor. */
  ~PatternIntensityMetric() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  PatternIntensityMetric(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxPatternIntensityMetric.hxx"
#endif

#endif // end #ifndef elxPatternIntensityMetric_h
