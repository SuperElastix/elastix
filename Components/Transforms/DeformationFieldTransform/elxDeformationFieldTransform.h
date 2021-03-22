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
  /** Standard ITK-stuff. */
  typedef DeformationFieldTransform Self;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  typedef itk::DeformationFieldInterpolatingTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                      elx::TransformBase<TElastix>::FixedImageDimension,
                                                      float>
    DeformationFieldInterpolatingTransformType;

  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
    Superclass1;

  typedef elx::TransformBase<TElastix> Superclass2;

  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;

  /** Typedef's specific for the DeformationFieldInterpolatingTransform. */
  typedef typename DeformationFieldInterpolatingTransformType::DeformationFieldType       DeformationFieldType;
  typedef typename DeformationFieldInterpolatingTransformType::DeformationFieldVectorType DeformationFieldVectorType;

  typedef typename DeformationFieldInterpolatingTransformType::Pointer DeformationFieldInterpolatingTransformPointer;

  /** Typedef's from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ParameterMapType         ParameterMapType;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

protected:
  /** The constructor. */
  DeformationFieldTransform();
  /** The destructor. */
  ~DeformationFieldTransform() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** Writes its deformation field to a file. */
  void
  WriteDerivedTransformDataToFile(void) const override;

  /** The deleted copy constructor. */
  DeformationFieldTransform(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  typedef typename DeformationFieldType::DirectionType DirectionType;

  /** The transform that is set as current transform in the
   * CcombinationTransform */
  DeformationFieldInterpolatingTransformPointer m_DeformationFieldInterpolatingTransform;

  /** Original direction cosines; stored to facilitate UseDirectionCosines option. */
  DirectionType m_OriginalDeformationFieldDirection;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxDeformationFieldTransform.hxx"
#endif

#endif // end #ifndef elxDeformationFieldTransform_h
