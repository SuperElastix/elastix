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
#ifndef elxRayCastInterpolator_h
#define elxRayCastInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTransform.h"
#include "itkEulerTransform.h"

namespace elastix
{

/**
 * \class RayCastInterpolator
 * \brief An interpolator based on the itkAdvancedRayCastInterpolateImageFunction.
 *
 *
 *
 * The parameters used in this class are:
 * \parameter Interpolator: Select this interpolator as follows:\n
 *    <tt>(Interpolator "RayCastInterpolator")</tt>
 *
 * \ingroup Interpolators
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT RayCastInterpolator
  : public itk::AdvancedRayCastInterpolateImageFunction<typename InterpolatorBase<TElastix>::InputImageType,
                                                        typename InterpolatorBase<TElastix>::CoordRepType>
  , public InterpolatorBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RayCastInterpolator);

  /** Standard ITK-stuff. */
  using Self = RayCastInterpolator;
  using Superclass1 = itk::AdvancedRayCastInterpolateImageFunction<typename InterpolatorBase<TElastix>::InputImageType,
                                                                   typename InterpolatorBase<TElastix>::CoordRepType>;
  using Superclass2 = InterpolatorBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayCastInterpolator, AdvancedRayCastInterpolateImageFunction);

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(Interpolator "RayCastInterpolator")</tt>\n
   */
  elxClassNameMacro("RayCastInterpolator");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::OutputType;
  using typename Superclass1::InputImageType;
  using typename Superclass1::IndexType;
  using typename Superclass1::ContinuousIndexType;
  using typename Superclass1::PointType;
  using typename Superclass1::SizeType;
  using SpacingType = typename InputImageType::SpacingType;

  /** Typedefs inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Typedef's for CombinationTransform */
  using EulerTransformType =
    typename itk::EulerTransform<typename InterpolatorBase<TElastix>::CoordRepType, ImageDimension>;
  using TransformParametersType = typename EulerTransformType::ParametersType;
  using EulerTransformPointer = typename EulerTransformType::Pointer;
  using AdvancedTransformType = typename itk::
    AdvancedTransform<typename InterpolatorBase<TElastix>::CoordRepType, Self::ImageDimension, Self::ImageDimension>;
  using AdvancedTransformPointer = typename AdvancedTransformType::Pointer;
  using CombinationTransformType =
    typename itk::AdvancedCombinationTransform<typename InterpolatorBase<TElastix>::CoordRepType, Self::ImageDimension>;
  using CombinationTransformPointer = typename CombinationTransformType::Pointer;

protected:
  /** The constructor. */
  RayCastInterpolator() = default;

  /** The destructor. */
  ~RayCastInterpolator() override = default;

  int
  BeforeAll() override;

  void
  BeforeRegistration() override;

  void
  BeforeEachResolution() override;

private:
  elxOverrideGetSelfMacro;

  EulerTransformPointer       m_PreTransform;
  TransformParametersType     m_PreParameters;
  CombinationTransformPointer m_CombinationTransform;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxRayCastInterpolator.hxx"
#endif

#endif // end #ifndef elxRayCastInterpolator_h
