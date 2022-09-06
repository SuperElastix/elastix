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
#ifndef elxMultiResolutionRegistrationWithFeatures_h
#define elxMultiResolutionRegistrationWithFeatures_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiResolutionImageRegistrationMethodWithFeatures.h"

namespace elastix
{

/**
 * \class MultiResolutionRegistrationWithFeatures
 * \brief A registration framework based on the
 *   itk::MultiResolutionImageRegistrationMethodWithFeatures.
 *
 * This MultiResolutionRegistrationWithFeatures gives a framework for registration with a
 * multi-resolution approach, using ...
 * Like this for example:\n
 * <tt>(Interpolator "BSplineInterpolator" "BSplineInterpolator")</tt>
 *
 *
 * The parameters used in this class are:\n
 * \parameter Registration: Select this registration framework as follows:\n
 *    <tt>(Registration "MultiResolutionRegistrationWithFeatures")</tt>
 * \parameter NumberOfResolutions: the number of resolutions used. \n
 *    example: <tt>(NumberOfResolutions 4)</tt> \n
 *    The default is 3.\n
 * \parameter Metric\<i\>Weight: The weight for the i-th metric, in each resolution \n
 *    example: <tt>(Metric0Weight 0.5 0.5 0.8)</tt> \n
 *    example: <tt>(Metric1Weight 0.5 0.5 0.2)</tt> \n
 *    The default is 1.0.
 *
 * \ingroup Registrations
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MultiResolutionRegistrationWithFeatures
  : public itk::MultiResolutionImageRegistrationMethodWithFeatures<typename RegistrationBase<TElastix>::FixedImageType,
                                                                   typename RegistrationBase<TElastix>::MovingImageType>
  , public RegistrationBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiResolutionRegistrationWithFeatures);

  /** Standard ITK: Self */
  using Self = MultiResolutionRegistrationWithFeatures;
  using Superclass1 =
    itk::MultiResolutionImageRegistrationMethodWithFeatures<typename RegistrationBase<TElastix>::FixedImageType,
                                                            typename RegistrationBase<TElastix>::MovingImageType>;
  using Superclass2 = RegistrationBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionRegistrationWithFeatures, MultiResolutionImageRegistrationMethodWithFeatures);

  /** Name of this class.
   * Use this name in the parameter file to select this specific registration framework. \n
   * example: <tt>(Registration "MultiResolutionRegistrationWithFeatures")</tt>\n
   */
  elxClassNameMacro("MultiResolutionRegistrationWithFeatures");

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

  /** The CombinationMetric type, which is used internally by the Superclass1 */
  // using typename Superclass1::CombinationMetricType;
  // using typename Superclass1::CombinationMetricPointer;

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

protected:
  /** The constructor. */
  MultiResolutionRegistrationWithFeatures() = default;

  /** The destructor. */
  ~MultiResolutionRegistrationWithFeatures() override = default;

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
  GetAndSetComponents();

  /** Set the fixed image regions. */
  virtual void
  GetAndSetFixedImageRegions();

  /** Create and set the fixed image interpolators. */
  virtual void
  GetAndSetFixedImageInterpolators();

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiResolutionRegistrationWithFeatures.hxx"
#endif

#endif // end #ifndef elxMultiResolutionRegistrationWithFeatures_h
