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
#ifndef elxDeformationFieldTransform_h
#define elxDeformationFieldTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkDeformationFieldInterpolatingTransform.h"
#include "itkAdvancedCombinationTransform.h"

namespace elastix
{

/**
 * \class DeformationFieldTransform
 * \brief A transform based on a DeformationField.
 *
 * This transform models the transformation by a deformation vector field.
 * This transform is NOT meant to be used for optimisation. Just use it as an initial
 * transform, or with transformix.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "DeformationFieldTransform")</tt>
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter DeformationFieldFileName: stores the name of the deformation field. \n
 *    example: <tt>(DeformationFieldFileName "defField.mhd")</tt>
 * \transformparameter DeformationFieldInterpolationOrder: The interpolation order used for interpolating the
 * deformation field:\n example: <tt>(DeformationFieldInterpolationOrder 0)</tt>\n The default value is 0. Choose from
 * the allowed values 0 or 1.
 *
 *
 * \sa DeformationFieldInterpolatingTransform
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT DeformationFieldTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DeformationFieldTransform);

  /** Standard ITK-stuff. */
  using Self = DeformationFieldTransform;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using DeformationFieldInterpolatingTransformType =
    itk::DeformationFieldInterpolatingTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                elx::TransformBase<TElastix>::FixedImageDimension,
                                                float>;

  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;

  using Superclass2 = elx::TransformBase<TElastix>;

  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DeformationFieldTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "DeformationFieldTransform")</tt>\n
   */
  elxClassNameMacro("DeformationFieldTransform");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ScalarType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::NumberOfParametersType;
  using typename Superclass1::JacobianType;
  using typename Superclass1::InputVectorType;
  using typename Superclass1::OutputVectorType;
  using typename Superclass1::InputCovariantVectorType;
  using typename Superclass1::OutputCovariantVectorType;
  using typename Superclass1::InputVnlVectorType;
  using typename Superclass1::OutputVnlVectorType;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;

  /** Typedef's specific for the DeformationFieldInterpolatingTransform. */
  using DeformationFieldType = typename DeformationFieldInterpolatingTransformType::DeformationFieldType;
  using DeformationFieldVectorType = typename DeformationFieldInterpolatingTransformType::DeformationFieldVectorType;

  using DeformationFieldInterpolatingTransformPointer = typename DeformationFieldInterpolatingTransformType::Pointer;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  DeformationFieldTransform();
  /** The destructor. */
  ~DeformationFieldTransform() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** Writes its deformation field to a file. */
  void
  WriteDerivedTransformDataToFile() const override;

  using DirectionType = typename DeformationFieldType::DirectionType;

  /** The transform that is set as current transform in the
   * CcombinationTransform */
  const DeformationFieldInterpolatingTransformPointer m_DeformationFieldInterpolatingTransform{
    DeformationFieldInterpolatingTransformType::New()
  };

  /** Original direction cosines; stored to facilitate UseDirectionCosines option. */
  DirectionType m_OriginalDeformationFieldDirection;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxDeformationFieldTransform.hxx"
#endif

#endif // end #ifndef elxDeformationFieldTransform_h
