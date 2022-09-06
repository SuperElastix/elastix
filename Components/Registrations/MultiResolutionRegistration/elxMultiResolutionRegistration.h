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
#ifndef elxMultiResolutionRegistration_h
#define elxMultiResolutionRegistration_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkMultiResolutionImageRegistrationMethod2.h"

namespace elastix
{

/**
 * \class MultiResolutionRegistration
 * \brief A registration framework based on the itk::MultiResolutionImageRegistrationMethod.
 *
 * This MultiResolutionRegistration gives a framework for registration with a
 * multi-resolution approach.
 *
 * The parameters used in this class are:
 * \parameter Registration: Select this registration framework as follows:\n
 *    <tt>(Registration "MultiResolutionRegistration")</tt>
 * \parameter NumberOfResolutions: the number of resolutions used. \n
 *    example: <tt>(NumberOfResolutions 4)</tt> \n
 *    The default is 3.
 *
 * \ingroup Registrations
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MultiResolutionRegistration
  : public RegistrationBase<TElastix>::ITKBaseType
  , public RegistrationBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiResolutionRegistration);

  /** Standard ITK. */
  using Self = MultiResolutionRegistration;
  using Superclass1 = typename RegistrationBase<TElastix>::ITKBaseType;
  using Superclass2 = RegistrationBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionRegistration, MultiResolutionImageRegistrationMethod);

  /** Name of this class.
   * Use this name in the parameter file to select this specific registration framework. \n
   * example: <tt>(Registration "MultiResolutionRegistration")</tt>\n
   */
  elxClassNameMacro("MultiResolutionRegistration");

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
   * \li Set the fixed image region. */
  void
  BeforeRegistration() override;

  /** Execute stuff before each resolution:
   * \li Update masks with an erosion. */
  void
  BeforeEachResolution() override;

protected:
  /** The constructor. */
  MultiResolutionRegistration() = default;
  /** The destructor. */
  ~MultiResolutionRegistration() override = default;

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

  /** Function to update fixed and moving masks. */
  void
  UpdateMasks(unsigned int level);

  /** Read the components from m_Elastix and set them in the Registration class. */
  virtual void
  SetComponents();

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiResolutionRegistration.hxx"
#endif

#endif // end #ifndef elxMultiResolutionRegistration_h
