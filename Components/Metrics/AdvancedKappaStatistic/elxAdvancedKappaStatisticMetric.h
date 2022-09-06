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
#ifndef elxAdvancedKappaStatisticMetric_h
#define elxAdvancedKappaStatisticMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedKappaStatisticImageToImageMetric.h"

namespace elastix
{

/**
 * \class AdvancedKappaStatisticMetric
 * \brief An metric based on the itk::AdvancedKappaStatisticImageToImageMetric.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "AdvancedKappaStatistic")</tt>
 * \parameter UseComplement: Bool to use the complement of the metric or not.\n
 *    If true, the 1 - KappaStatistic is returned, which is useful since most
 *    optimizers search by default for a minimum.\n
 *    <tt>(UseComplement "true")</tt>\n
 *    The default value is true.
 * \parameter ForeGroundvalue: the overlap of structures with this value is
 *    calculated.\n
 *    <tt>(ForeGroundvalue 3.5)</tt>\n
 *    The default value is 1.0.
 *
 * \ingroup Metrics
 *
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT AdvancedKappaStatisticMetric
  : public itk::AdvancedKappaStatisticImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                         typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedKappaStatisticMetric);

  /** Standard ITK-stuff. */
  using Self = AdvancedKappaStatisticMetric;
  using Superclass1 = itk::AdvancedKappaStatisticImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                    typename MetricBase<TElastix>::MovingImageType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedKappaStatisticMetric, itk::AdvancedKappaStatisticImageToImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "AdvancedKappaStatistic")</tt>\n
   */
  elxClassNameMacro("AdvancedKappaStatistic");

  /** Typedefs from the superclass. */
  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImagePixelType;
  using typename Superclass1::MovingImageConstPointer;
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::TransformParametersType;
  using typename Superclass1::TransformJacobianType;
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::InterpolatorPointer;
  using typename Superclass1::RealType;
  using typename Superclass1::GradientPixelType;
  using typename Superclass1::GradientImageType;
  using typename Superclass1::GradientImagePointer;
  using typename Superclass1::GradientImageFilterType;
  using typename Superclass1::GradientImageFilterPointer;
  using typename Superclass1::FixedImageMaskType;
  using typename Superclass1::FixedImageMaskPointer;
  using typename Superclass1::MovingImageMaskType;
  using typename Superclass1::MovingImageMaskPointer;
  using typename Superclass1::MeasureType;
  using typename Superclass1::DerivativeType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::FixedImagePixelType;
  using typename Superclass1::MovingImageRegionType;
  using typename Superclass1::ImageSamplerType;
  using typename Superclass1::ImageSamplerPointer;
  using typename Superclass1::ImageSampleContainerType;
  using typename Superclass1::ImageSampleContainerPointer;
  using typename Superclass1::FixedImageLimiterType;
  using typename Superclass1::MovingImageLimiterType;
  using typename Superclass1::FixedImageLimiterOutputType;
  using typename Superclass1::MovingImageLimiterOutputType;
  using typename Superclass1::MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before registration:
   * \li Set the UseComplement setting
   * \li Set the ForeGroundvalue setting
   */
  void
  BeforeRegistration() override;

protected:
  /** The constructor. */
  AdvancedKappaStatisticMetric() = default;
  /** The destructor. */
  ~AdvancedKappaStatisticMetric() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxAdvancedKappaStatisticMetric.hxx"
#endif

#endif // end #ifndef elxAdvancedKappaStatisticMetric_h
