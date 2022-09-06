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
#ifndef elxMultiMetricMultiResolutionRegistration_h
#define elxMultiMetricMultiResolutionRegistration_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiMetricMultiResolutionImageRegistrationMethod.h"

namespace elastix
{

/**
 * \class MultiMetricMultiResolutionRegistration
 * \brief A registration framework based on the
 * itk::MultiMetricMultiResolutionImageRegistrationMethod.
 *
 * This MultiMetricMultiResolutionRegistration gives a framework for
 * registration with a multi-resolution approach, using multiple metrics.
 * The metrics can use the same pair of images/image pyramids/interpolators
 * /masks, but also different pairs of fixed/moving images etc.
 * If the metrics use the same moving image and the same moving image
 * pyramid, they can use the same interpolator. If multiple moving images
 * are used, enter multiple interpolators, possibly  of the same type:
 * Like this for example:\n
 * <tt>(Interpolator "BSplineInterpolator" "BSplineInterpolator")</tt>\n
 * For each metric a different instance of an image sampler can be used:\n
 * <tt>(ImageSampler "Random" "Random")</tt>\n
 * or:\n
 * <tt>(ImageSampler "Random" "Full")</tt>\n
 *
 * Note, that the number of metrics should always be larger than or equal
 * to the number of fixed/moving images, interpolators, image pyramids etc.
 * Also, when all metrics need an image sampler, for each fixed image pyramid,
 * an image sampler must be provided. In some cases, one sampler can be
 * used for all metrics. This is the case when multiple metrics are desired,
 * but
 * \li 1 fixed image is used, and
 * \li 1 fixed image pyramid is used.
 * This will save a bit of memory and computation time.
 * In general however, it is better to use the same number of samplers as
 * metrics.
 *
 * The parameters used in this class are:\n
 * \parameter Registration: Select this registration framework as follows:\n
 *    <tt>(Registration "MultiMetricMultiResolutionRegistration")</tt>
 * \parameter NumberOfResolutions: the number of resolutions used. \n
 *    example: <tt>(NumberOfResolutions 4)</tt> \n
 *    The default is 3.\n
 * \parameter Metric\<i\>Weight: The weight for the i-th metric,
 *    in each resolution. \n
 *    example: <tt>(Metric0Weight 0.5 0.5 0.8)</tt> \n
 *    example: <tt>(Metric1Weight 0.5 0.5 0.2)</tt> \n
 *    The default is 1 / numberOfMetrics.
 * \parameter Metric\<i\>RelativeWeight: The relative weight \f$rw_i\f$
 *    for the i-th metric, in each resolution. \n
 *    This is an alternative to the default Metric\<i\>Weight and can
 *    be selected with the parameter UseRelativeWeights. The weight \f$w_i\f$
 *    is computed in each iteration based on the magnitude of the metric
 *    derivative (gradient) \f$|g_i|\f$ related to that of the first metric
 *    \f$|g_0|\f$. It is done such that the fraction \f$|g_0|/|g_i|\f$ is
 *    identical to \f$rw_0/rw_i\f$, so
 *      \f[w_i = rw_i |g_0|/|g_i|.\f]
 *    example: <tt>(Metric0RelativeWeight 0.5 0.5 0.8)</tt> \n
 *    example: <tt>(Metric1RelativeWeight 0.5 0.5 0.2)</tt> \n
 *    The default is 1 / numberOfMetrics.
 * \parameter UseRelativeWeights: Whether relative weights are used
 *    or simple static, fixed weights. \n
 *    example: <tt>(UseRelativeWeights "false" "true")</tt> \n
 *    The default is "false", which means using Metric\<i\>Weight.
 * \parameter Metric\<i\>Use: Whether the i-th metric is only computed or
 *    also used, in each resolution. \n
 *    example: <tt>(Metric0Use "false" "true")</tt> \n
 *    example: <tt>(Metric1Use "true" "false")</tt> \n
 *    The default is "true".
 *
 * \ingroup Registrations
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MultiMetricMultiResolutionRegistration
  : public itk::MultiMetricMultiResolutionImageRegistrationMethod<typename RegistrationBase<TElastix>::FixedImageType,
                                                                  typename RegistrationBase<TElastix>::MovingImageType>
  , public RegistrationBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiMetricMultiResolutionRegistration);

  /** Standard ITK: Self */
  using Self = MultiMetricMultiResolutionRegistration;

  /** Standard ITK: Superclasses. */
  using Superclass1 =
    itk::MultiMetricMultiResolutionImageRegistrationMethod<typename RegistrationBase<TElastix>::FixedImageType,
                                                           typename RegistrationBase<TElastix>::MovingImageType>;
  using Superclass2 = RegistrationBase<TElastix>;

  /** Standard ITK: SmartPointers */
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiMetricMultiResolutionRegistration, MultiMetricMultiResolutionImageRegistrationMethod);

  /** Name of this class.
   * Use this name in the parameter file to select this specific registration framework. \n
   * example: <tt>(Registration "MultiMetricMultiResolutionRegistration")</tt>\n
   */
  elxClassNameMacro("MultiMetricMultiResolutionRegistration");

  /** Typedef's inherited from Superclass1. */

  /**  Type of the Fixed image. */
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;

  /**  Type of the Moving image. */
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImageConstPointer;

  /**  Type of the metric. */
  using typename Superclass1::MetricType;
  using typename Superclass1::MetricPointer;

  /**  Type of the Transform . */
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;

  /**  Type of the Interpolator. */
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::InterpolatorPointer;

  /**  Type of the optimizer. */
  using typename Superclass1::OptimizerType;
  using typename Superclass1::OptimizerPointer;

  /** Type of the Fixed image multiresolution pyramid. */
  using typename Superclass1::FixedImagePyramidType;
  using typename Superclass1::FixedImagePyramidPointer;

  /** Type of the moving image multiresolution pyramid. */
  using typename Superclass1::MovingImagePyramidType;
  using typename Superclass1::MovingImagePyramidPointer;

  /** Type of the Transformation parameters. This is the same type used to
   *  represent the search space of the optimization algorithm.
   */
  using typename Superclass1::ParametersType;

  /** The CombinationMetric type, which is used internally by the Superclass1. */
  using typename Superclass1::CombinationMetricType;
  using typename Superclass1::CombinationMetricPointer;

  /** Typedef's from Elastix. */
  using typename Superclass2::ElastixType;
  using RegistrationType = typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using typename Superclass2::UseMaskErosionArrayType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, Superclass2::FixedImageDimension);
  /** Get the dimension of the moving image. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, Superclass2::MovingImageDimension);

  /** Execute stuff before the actual registration:
   * \li Connect all components to the registration framework.
   * \li Set the number of resolution levels.
   * \li Set the fixed image regions.
   * \li Add the sub metric columns to the iteration info object.
   */
  void
  BeforeRegistration() override;

  /** Execute stuff before each resolution:
   * \li Update masks with an erosion.
   * \li Set the metric weights.
   */
  void
  BeforeEachResolution() override;

  /** Execute stuff after each iteration
   * \li Print the latest computed submetric values to screen.
   */
  void
  AfterEachIteration() override;

protected:
  /** The constructor. */
  MultiMetricMultiResolutionRegistration();
  /** The destructor. */
  ~MultiMetricMultiResolutionRegistration() override = default;

  /** Typedef's for mask support. */
  using typename Superclass2::MaskPixelType;
  using typename Superclass2::FixedMaskImageType;
  using typename Superclass2::MovingMaskImageType;
  using typename Superclass2::FixedMaskImagePointer;
  using typename Superclass2::MovingMaskImagePointer;
  using typename Superclass2::FixedMaskSpatialObjectType;
  using typename Superclass2::MovingMaskSpatialObjectType;
  using typename Superclass2::FixedMaskSpatialObjectPointer;
  using typename Superclass2::MovingMaskSpatialObjectPointer;

  /** Function to update masks. */
  void
  UpdateFixedMasks(unsigned int level);

  void
  UpdateMovingMasks(unsigned int level);

  /** Read the components from m_Elastix and set them in the Registration class. */
  virtual void
  SetComponents();

  bool m_ShowExactMetricValue;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiMetricMultiResolutionRegistration.hxx"
#endif

#endif // end #ifndef elxMultiMetricMultiResolutionRegistration_h
