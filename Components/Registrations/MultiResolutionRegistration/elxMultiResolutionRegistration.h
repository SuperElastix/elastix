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
  /** Standard ITK. */
  typedef MultiResolutionRegistration                      Self;
  typedef typename RegistrationBase<TElastix>::ITKBaseType Superclass1;
  typedef RegistrationBase<TElastix>                       Superclass2;
  typedef itk::SmartPointer<Self>                          Pointer;
  typedef itk::SmartPointer<const Self>                    ConstPointer;

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
  typedef typename Superclass1::OptimizerType OptimizerType;

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
   * \li Set the fixed image region. */
  void
  BeforeRegistration(void) override;

  /** Execute stuff before each resolution:
   * \li Update masks with an erosion. */
  void
  BeforeEachResolution(void) override;

protected:
  /** The constructor. */
  MultiResolutionRegistration() = default;
  /** The destructor. */
  ~MultiResolutionRegistration() override = default;

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

  /** Function to update fixed and moving masks. */
  void
  UpdateMasks(unsigned int level);

  /** Read the components from m_Elastix and set them in the Registration class. */
  virtual void
  SetComponents(void);

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  MultiResolutionRegistration(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiResolutionRegistration.hxx"
#endif

#endif // end #ifndef elxMultiResolutionRegistration_h
