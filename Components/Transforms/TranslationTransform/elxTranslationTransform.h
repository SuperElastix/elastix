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
#ifndef elxTranslationTransform_h
#define elxTranslationTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedTranslationTransform.h"
#include "itkTranslationTransformInitializer.h"

namespace elastix
{

/**
 * \class TranslationTransformElastix
 * \brief A transform based on the itk::TranslationTransform.
 *
 * This transform is a translation transformation.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "TranslationTransform")</tt>
 * \parameter AutomaticTransformInitialization: whether or not the initial translation
 *    between images should be estimated as the distance between their centers.\n
 *    example: <tt>(AutomaticTransformInitialization "true")</tt> \n
 *    By default "false" is assumed. So, no initial translation.
 * \parameter AutomaticTransformInitializationMethod: how to initialize this
 *    transform. Should be one of {GeometricalCenter, CenterOfGravity}.\n
 *    example: <tt>(AutomaticTransformInitializationMethod "CenterOfGravity")</tt> \n
 *    By default "GeometricalCenter" is assumed.\n
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT TranslationTransformElastix
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TranslationTransformElastix);

  /** Standard ITK-stuff. */
  using Self = TranslationTransformElastix;

  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;

  using Superclass2 = elx::TransformBase<TElastix>;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  using TranslationTransformType =
    itk::AdvancedTranslationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                      elx::TransformBase<TElastix>::FixedImageDimension>;

  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslationTransformElastix, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "TranslationTransform")</tt>\n
   */
  elxClassNameMacro("TranslationTransform");

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ScalarType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::JacobianType;
  using typename Superclass1::InputVectorType;
  using typename Superclass1::OutputVectorType;
  using typename Superclass1::InputCovariantVectorType;
  using typename Superclass1::OutputCovariantVectorType;
  using typename Superclass1::InputVnlVectorType;
  using typename Superclass1::OutputVnlVectorType;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::NumberOfParametersType;

  /** Typedef's from the TransformBase class. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Extra typedefs */
  using TransformInitializerType =
    itk::TranslationTransformInitializer<TranslationTransformType, FixedImageType, MovingImageType>;
  using TransformInitializerPointer = typename TransformInitializerType::Pointer;
  using TranslationTransformPointer = typename TranslationTransformType::Pointer;

  /** Execute stuff before the actual registration:
   * \li Call InitializeTransform.
   */
  void
  BeforeRegistration() override;

protected:
  /** The constructor. */
  TranslationTransformElastix();
  /** The destructor. */
  ~TranslationTransformElastix() override = default;

  const TranslationTransformPointer m_TranslationTransform{ TranslationTransformType::New() };

private:
  elxOverrideGetSelfMacro;

  /** Initialize Transform.
   * \li Set all parameters to zero.
   * \li Set initial translation:
   *  the initial translation between fixed and moving image is guessed,
   *  if the user has set (AutomaticTransformInitialization "true").
   */
  void
  InitializeTransform();

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTranslationTransform.hxx"
#endif

#endif // end #ifndef elxTranslationTransform_h
