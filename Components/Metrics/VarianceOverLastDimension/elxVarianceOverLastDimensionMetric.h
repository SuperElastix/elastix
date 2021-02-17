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
#ifndef elxVarianceOverLastDimensionMetric_h
#define elxVarianceOverLastDimensionMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkVarianceOverLastDimensionImageMetric.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkStackTransform.h"

namespace elastix
{

/** \class VarianceOverLastDimensionMetric
 * \brief Compute the sum of variances over the slowest varying dimension in the moving image.
 *
 * For a description of this metric see the paper:\n
 * <em>Nonrigid registration of dynamic medical imaging data using
 * nD+t B-splines and a groupwise optimization approach</em>,
 * C.T. Metz, S. Klein, M. Schaap, T. van Walsum and W.J. Niessen,
 * Medical Image Analysis, in press.
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
class ITK_TEMPLATE_EXPORT VarianceOverLastDimensionMetric
  : public itk::VarianceOverLastDimensionImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                     typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef VarianceOverLastDimensionMetric Self;
  typedef itk::VarianceOverLastDimensionImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                    typename MetricBase<TElastix>::MovingImageType>
                                        Superclass1;
  typedef MetricBase<TElastix>          Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VarianceOverLastDimensionMetric, itk::VarianceOverLastDimensionImageMetric);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "VarianceOverLastDimensionMetric")</tt>\n
   */
  elxClassNameMacro("VarianceOverLastDimensionMetric");

  /** Typedefs from the superclass. */
  typedef typename Superclass1::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass1::ScalarType                      ScalarType;
  typedef typename Superclass1::MovingImageType                 MovingImageType;
  typedef typename Superclass1::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass1::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass1::FixedImageType                  FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer          FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType            FixedImageRegionType;
  typedef typename Superclass1::FixedImageSizeType              FixedImageSizeType;
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

  /** Typedef's for the B-spline transform. */
  typedef itk::AdvancedBSplineDeformableTransformBase<ScalarType, FixedImageDimension> BSplineTransformBaseType;
  typedef itk::AdvancedCombinationTransform<ScalarType, FixedImageDimension>           CombinationTransformType;
  typedef itk::StackTransform<ScalarType, FixedImageDimension, MovingImageDimension>   StackTransformType;
  typedef itk::AdvancedBSplineDeformableTransformBase<ScalarType, FixedImageDimension - 1>
    ReducedDimensionBSplineTransformBaseType;

  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize(void) override;

  /**
   * Do some things before registration:
   * \li check the direction cosines
   */
  void
  BeforeRegistration(void) override;

  /**
   * Do some things before each resolution:
   * \li Set CheckNumberOfSamples setting
   * \li Set UseNormalization setting
   */
  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  VarianceOverLastDimensionMetric() = default;
  /** The destructor. */
  ~VarianceOverLastDimensionMetric() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  VarianceOverLastDimensionMetric(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxVarianceOverLastDimensionMetric.hxx"
#endif

#endif // end #ifndef elxVarianceOverLastDimensionMetric_h
