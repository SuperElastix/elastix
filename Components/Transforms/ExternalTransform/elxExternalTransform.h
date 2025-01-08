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
#ifndef elxExternalTransform_h
#define elxExternalTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "elxAdvancedTransformAdapter.h"
#include "elxDefaultConstruct.h"
#include "itkAdvancedCombinationTransform.h"

namespace elastix
{

/**
 * \class ExternalTransform
 * \brief An external transform.
 *
 * This transform warps an external transform.
 * This transform is NOT meant to be used for optimization. Just use it as an initial transform, or with transformix.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "ExternalTransform")</tt>
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter TransformAddress: specifies the memory address of the external transform. \n
 *    example: <tt>(TransformAddress "000002A3CB9AACC0")</tt>
 *
 * \ingroup Transforms
 */

template <typename TElastix>
class ITK_TEMPLATE_EXPORT ExternalTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordinateType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ExternalTransform);

  /** Standard ITK-stuff. */
  using Self = ExternalTransform;

  /** The ITK-class that provides most of the functionality */
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordinateType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;

  using Superclass2 = elx::TransformBase<TElastix>;

  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::CoordinateType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ExternalTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "ExternalTransform")</tt>\n
   */
  elxClassNameMacro("ExternalTransform");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** Default-constructor. */
  ExternalTransform();

  /** Destructor. */
  ~ExternalTransform() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParameterMap() const override;

  using AdvancedTransformAdapterType = AdvancedTransformAdapter<CoordinateType, Superclass2::FixedImageDimension>;

  /** The transform that is set as current transform in the CombinationTransform */
  const itk::SmartPointer<AdvancedTransformAdapterType> m_AdvancedTransformAdapter{
    AdvancedTransformAdapterType::New()
  };
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxExternalTransform.hxx"
#endif

#endif