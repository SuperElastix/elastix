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
  /** Standard ITK-stuff. */
  typedef TranslationTransformElastix Self;

  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
    Superclass1;

  typedef elx::TransformBase<TElastix> Superclass2;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  typedef itk::AdvancedTranslationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
    TranslationTransformType;

  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;

  /** Typedef's from the TransformBase class. */
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

  /** Extra typedefs */
  typedef itk::TranslationTransformInitializer<TranslationTransformType, FixedImageType, MovingImageType>
                                                     TransformInitializerType;
  typedef typename TransformInitializerType::Pointer TransformInitializerPointer;
  typedef typename TranslationTransformType::Pointer TranslationTransformPointer;

  /** Execute stuff before the actual registration:
   * \li Call InitializeTransform.
   */
  void
  BeforeRegistration(void) override;

  /** Initialize Transform.
   * \li Set all parameters to zero.
   * \li Set initial translation:
   *  the initial translation between fixed and moving image is guessed,
   *  if the user has set (AutomaticTransformInitialization "true").
   */
  virtual void
  InitializeTransform(void);

protected:
  /** The constructor. */
  TranslationTransformElastix();
  /** The destructor. */
  ~TranslationTransformElastix() override = default;

  TranslationTransformPointer m_TranslationTransform;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  TranslationTransformElastix(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTranslationTransform.hxx"
#endif

#endif // end #ifndef elxTranslationTransform_h
