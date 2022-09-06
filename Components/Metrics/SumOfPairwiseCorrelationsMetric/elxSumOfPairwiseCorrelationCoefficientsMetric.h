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
#ifndef elxSumOfPairwiseCorrelationCoefficientsMetric_h
#define elxSumOfPairwiseCorrelationCoefficientsMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkSumOfPairwiseCorrelationCoefficientsMetric.h"

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkStackTransform.h"

namespace elastix
{
/** \class SumOfPairwiseCorrelationCoefficientsMetric
 * \brief Compute the sum of correlation coefficients between images in the
 *  slowest varying dimension in the moving image.
 *
 *
 * This metric is based on the AdvancedImageToImageMetric.
 * It is templated over the type of the fixed and moving images to be compared.
 *
 * This metric computes the sum of variances over the slowest varying dimension in
 * the moving image. The spatial positions of the moving image are established
 * through a Transform. Pixel values are taken from the Moving image.
 *
 * This implementation is based on the AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * \parameter SampleLastDimensionRandomly: randomly sample a number of time points to
 *    to compute the variance from. When set to "false", all time points are taken into
 *    account. When set to "true", a random number of time points is selected, which can
 *    be set with parameter NumSamplesLastDimension. \n
 * \parameter NumSamplesLastDimension: the number of random samples to take in the time
 *    time direction of the data when SampleLastDimensionRandomly is set to true.
 * \parameter SubtractMean: subtract the over time computed mean parameter value from
 *    each parameter. This should be used when registration is performed directly on the moving
 *    image, without using a fixed image. Possible values are "true" or "false".
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT SumOfPairwiseCorrelationCoefficientsMetric
  : public itk::SumOfPairwiseCorrelationCoefficientsMetric<typename MetricBase<TElastix>::FixedImageType,
                                                           typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SumOfPairwiseCorrelationCoefficientsMetric);

  /** Standard ITK-stuff. */
  using Self = SumOfPairwiseCorrelationCoefficientsMetric;
  using Superclass1 = itk::SumOfPairwiseCorrelationCoefficientsMetric<typename MetricBase<TElastix>::FixedImageType,
                                                                      typename MetricBase<TElastix>::MovingImageType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SumOfPairwiseCorrelationCoefficientsMetric, itk::SumOfPairwiseCorrelationCoefficientsMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "SumOfPairwiseCorrelationCoefficientsMetric")</tt>\n
   */
  elxClassNameMacro("SumOfPairwiseCorrelationCoefficientsMetric");

  /** Typedefs from the superclass. */
  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::ScalarType;
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImagePixelType;
  using typename Superclass1::MovingImageConstPointer;
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::FixedImageSizeType;
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

  /** Typedef's for the B-spline transform. */
  using BSplineTransformBaseType = itk::AdvancedBSplineDeformableTransformBase<ScalarType, FixedImageDimension>;
  using CombinationTransformType = itk::AdvancedCombinationTransform<ScalarType, FixedImageDimension>;
  using StackTransformType = itk::StackTransform<ScalarType, FixedImageDimension, MovingImageDimension>;
  using ReducedDimensionBSplineTransformBaseType =
    itk::AdvancedBSplineDeformableTransformBase<ScalarType, FixedImageDimension - 1>;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before each resolution:
   * \li Set CheckNumberOfSamples setting
   * \li Set UseNormalization setting
   */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  SumOfPairwiseCorrelationCoefficientsMetric() = default;

  /** The destructor. */
  ~SumOfPairwiseCorrelationCoefficientsMetric() override = default;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxSumOfPairwiseCorrelationCoefficientsMetric.hxx"
#endif

#endif // end #ifndef elxSumOfPairwiseCorrelationCoefficientsMetric_h
