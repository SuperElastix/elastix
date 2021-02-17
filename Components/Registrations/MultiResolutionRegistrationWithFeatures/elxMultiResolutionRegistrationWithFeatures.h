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
  /** Standard ITK: Self */
  typedef MultiResolutionRegistrationWithFeatures Self;
  typedef itk::MultiResolutionImageRegistrationMethodWithFeatures<typename RegistrationBase<TElastix>::FixedImageType,
                                                                  typename RegistrationBase<TElastix>::MovingImageType>
                                        Superclass1;
  typedef RegistrationBase<TElastix>    Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  typedef typename Superclass1::FixedImageType         FixedImageType;
  typedef typename Superclass1::FixedImageConstPointer FixedImageConstPointer;
  typedef typename Superclass1::FixedImageRegionType   FixedImageRegionType;

  /**  Type of the Moving image. */
  typedef typename Superclass1::MovingImageType         MovingImageType;
  typedef typename Superclass1::MovingImageConstPointer MovingImageConstPointer;

  /**  Type of the metric. */
  typedef typename Superclass1::MetricType    MetricType;
  typedef typename Superclass1::MetricPointer MetricPointer;

  /**  Type of the Transform . */
  typedef typename Superclass1::TransformType    TransformType;
  typedef typename Superclass1::TransformPointer TransformPointer;

  /**  Type of the Interpolator. */
  typedef typename Superclass1::InterpolatorType    InterpolatorType;
  typedef typename Superclass1::InterpolatorPointer InterpolatorPointer;

  /**  Type of the optimizer. */
  typedef typename Superclass1::OptimizerType    OptimizerType;
  typedef typename Superclass1::OptimizerPointer OptimizerPointer;

  /** Type of the Fixed image multiresolution pyramid. */
  typedef typename Superclass1::FixedImagePyramidType    FixedImagePyramidType;
  typedef typename Superclass1::FixedImagePyramidPointer FixedImagePyramidPointer;

  /** Type of the moving image multiresolution pyramid. */
  typedef typename Superclass1::MovingImagePyramidType    MovingImagePyramidType;
  typedef typename Superclass1::MovingImagePyramidPointer MovingImagePyramidPointer;

  /** Type of the Transformation parameters. This is the same type used to
   *  represent the search space of the optimization algorithm.
   */
  typedef typename Superclass1::ParametersType ParametersType;

  /** The CombinationMetric type, which is used internally by the Superclass1 */
  // typedef typename Superclass1::CombinationMetricType     CombinationMetricType;
  // typedef typename Superclass1::CombinationMetricPointer  CombinationMetricPointer;

  /** Typedef's from Elastix. */
  typedef typename Superclass2::ElastixType             ElastixType;
  typedef typename Superclass2::ElastixPointer          ElastixPointer;
  typedef typename Superclass2::ConfigurationType       ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer    ConfigurationPointer;
  typedef typename Superclass2::RegistrationType        RegistrationType;
  typedef typename Superclass2::RegistrationPointer     RegistrationPointer;
  typedef typename Superclass2::ITKBaseType             ITKBaseType;
  typedef typename Superclass2::UseMaskErosionArrayType UseMaskErosionArrayType;

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
  BeforeRegistration(void) override;

  /** Execute stuff before each resolution:
   * \li Update masks with an erosion.
   * \li Set the metric weights.
   */
  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  MultiResolutionRegistrationWithFeatures() = default;

  /** The destructor. */
  ~MultiResolutionRegistrationWithFeatures() override = default;

  /** Typedef's for mask support. */
  typedef typename Superclass2::MaskPixelType                  MaskPixelType;
  typedef typename Superclass2::FixedMaskImageType             FixedMaskImageType;
  typedef typename Superclass2::MovingMaskImageType            MovingMaskImageType;
  typedef typename Superclass2::FixedMaskImagePointer          FixedMaskImagePointer;
  typedef typename Superclass2::MovingMaskImagePointer         MovingMaskImagePointer;
  typedef typename Superclass2::FixedMaskSpatialObjectType     FixedMaskSpatialObjectType;
  typedef typename Superclass2::MovingMaskSpatialObjectType    MovingMaskSpatialObjectType;
  typedef typename Superclass2::FixedMaskSpatialObjectPointer  FixedMaskSpatialObjectPointer;
  typedef typename Superclass2::MovingMaskSpatialObjectPointer MovingMaskSpatialObjectPointer;

  /** Function to update masks. */
  void
  UpdateFixedMasks(unsigned int level);

  void
  UpdateMovingMasks(unsigned int level);

  /** Read the components from m_Elastix and set them in the Registration class. */
  virtual void
  GetAndSetComponents(void);

  /** Set the fixed image regions. */
  virtual void
  GetAndSetFixedImageRegions(void);

  /** Create and set the fixed image interpolators. */
  virtual void
  GetAndSetFixedImageInterpolators(void);

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  MultiResolutionRegistrationWithFeatures(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiResolutionRegistrationWithFeatures.hxx"
#endif

#endif // end #ifndef elxMultiResolutionRegistrationWithFeatures_h
