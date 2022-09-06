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
#ifndef elxNormalizedMutualInformationMetric_h
#define elxNormalizedMutualInformationMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkParzenWindowNormalizedMutualInformationImageToImageMetric.h"

namespace elastix
{

/**
 * \class NormalizedMutualInformationMetric
 * \brief A metric based on the itk::ParzenWindowNormalizedMutualInformationImageToImageMetric.
 *
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "NormalizedMutualInformation")</tt>
 * \parameter NumberOfHistogramBins: The size of the histogram. Must be given for each
 *    resolution, or for all resolutions at once. \n
 *    example: <tt>(NumberOfHistogramBins 32 32 64)</tt> \n
 *    The default is 32 for each resolution.
 * \parameter NumberOfFixedHistogramBins: The size of the histogram in the fixed dimension. Can be given for each
 *    resolution, or for all resolutions at once. If not given, NumberOfHistograms is used.\n
 *    example: <tt>(NumberOfFixedHistogramBins 32 32 64)</tt> \n
 *    The default is the value of NumberOfHistograms.
 * \parameter NumberOfMovingHistogramBins: The size of the histogram in the fixed dimension. Can be given for each
 *    resolution, or for all resolutions at once. If not given, NumberOfHistograms is used.\n
 *    example: <tt>(NumberOfMovingHistogramBins 32 32 64)</tt> \n
 *    The default is the value of NumberOfHistograms.
 * \parameter FixedKernelBSplineOrder: The B-spline order of the Parzen window, used to estimate
 *    the joint histogram. Can be given for each resolution, or for all resolutions at once. \n
 *    example: <tt>(FixedKernelBSplineOrder 0 1 1)</tt> \n
 *    The default value is 0.
 * \parameter MovingKernelBSplineOrder: The B-spline order of the Parzen window, used to estimate
 *    the joint histogram. Can be given for each resolution, or for all resolutions at once. \n
 *    example: <tt>(MovingKernelBSplineOrder 3 3 3)</tt> \n
 *    The default value is 3.
 * \parameter FixedLimitRangeRatio: The relative extension of the intensity range of the fixed image.\n
 *    If your image has gray values from 0 to 1000 and the FixedLimitRangeRatio is 0.001, the
 *    joint histogram will expect fixed image gray values from -0.001 to 1000.001. This may be
 *    useful if you use high order B-spline interpolator for the fixed image.\n
 *    example: <tt>(FixedLimitRangeRatio 0.001 0.01 0.01)</tt> \n
 *    The default value is 0.01. Can be given for each resolution, or for all resolutions at once.
 * \parameter MovingLimitRangeRatio: The relative extension of the intensity range of the moving image.\n
 *    If your image has gray values from 0 to 1000 and the MovingLimitRangeRatio is 0.001, the
 *    joint histogram will expect moving image gray values from -0.001 to 1000.001. This may be
 *    useful if you use high order B-spline interpolator for the moving image.\n
 *    example: <tt>(MovingLimitRangeRatio 0.001 0.01 0.01)</tt> \n
 *    The default value is 0.01. Can be given for each resolution, or for all resolutions at once.
 *
 * \sa ParzenWindowNormalizedMutualInformationImageToImageMetric
 * \ingroup Metrics
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT NormalizedMutualInformationMetric
  : public itk::ParzenWindowNormalizedMutualInformationImageToImageMetric<
      typename MetricBase<TElastix>::FixedImageType,
      typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(NormalizedMutualInformationMetric);

  /** Standard ITK-stuff. */
  using Self = NormalizedMutualInformationMetric;
  using Superclass1 =
    itk::ParzenWindowNormalizedMutualInformationImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                   typename MetricBase<TElastix>::MovingImageType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NormalizedMutualInformationMetric, itk::ParzenWindowNormalizedMutualInformationImageToImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "NormalizedMutualInformation")</tt>\n
   */
  elxClassNameMacro("NormalizedMutualInformation");

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

  /** Execute stuff before each new pyramid resolution:
   * \li Set the number of histogram bins.
   * \li Set the CheckNumberOfSamples option.
   * \li Set the fixed/moving LimitRangeRatio
   * \li Set the fixed/moving limiter. */
  void
  BeforeEachResolution() override;

  /** Set up a timer to measure the initialization time and
   * call the Superclass' implementation. */
  void
  Initialize() override;

protected:
  /** The constructor. */
  NormalizedMutualInformationMetric() { this->SetUseDerivative(true); }


  /** The destructor. */
  ~NormalizedMutualInformationMetric() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxNormalizedMutualInformationMetric.hxx"
#endif

#endif // end #ifndef elxNormalizedMutualInformationMetric_h
