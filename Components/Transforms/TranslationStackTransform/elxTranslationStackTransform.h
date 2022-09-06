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
#ifndef elxTranslationStackTransform_h
#define elxTranslationStackTransform_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTranslationTransform.h"
#include "itkTranslationStackTransform.h"


/**
 * \class TranslationStackTransform
 * \brief A Translation transform based on the itkStackTransform.
 *
 * This transform is a Translation transformation. Calls to TransformPoint and GetJacobian are
 * redirected to the appropriate sub transform based on the last dimension (time) index.
 *
 * This transform uses the size, spacing and origin of the last dimension of the fixed
 * image to set the number of sub transforms the origin of the first transform and the
 * spacing between the transforms.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "TranslationStackTransform")</tt>
 *
 * \transformparameter StackSpacing: stores the spacing between the sub transforms. \n
 *    exanoke: <tt>(StackSpacing 1.0)</tt>
 * \transformparameter StackOrigin: stores the origin of the first sub transform. \n
 *    exanoke: <tt>(StackOrigin 0.0)</tt>
 * \transformparameter NumberOfSubTransforms: stores the number of sub transforms. \n
 *    exanoke: <tt>(NumberOfSubTransforms 10)</tt>
 *
 * \todo It is unsure what happens when one of the image dimensions has length 1.
 *
 * \ingroup Transforms
 */

namespace elastix
{
template <class TElastix>
class ITK_TEMPLATE_EXPORT TranslationStackTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TranslationStackTransform);

  /** Standard ITK-stuff. */
  using Self = TranslationStackTransform;
  using Superclass1 = itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                                        elx::TransformBase<TElastix>::FixedImageDimension>;
  using Superclass2 = elx::TransformBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TranslationStackTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "TranslationStackTransform")</tt>\n
   */
  elxClassNameMacro("TranslationStackTransform");

  /** (Reduced) dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);
  itkStaticConstMacro(ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1);

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  using TranslationTransformType =
    itk::AdvancedTranslationTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::SpaceDimension>;
  using TranslationTransformPointer = typename TranslationTransformType::Pointer;

  /** The ITK-class for the sub transforms, which have a reduced dimension. */
  using ReducedDimensionTranslationTransformType =
    itk::AdvancedTranslationTransform<typename elx::TransformBase<TElastix>::CoordRepType, Self::ReducedSpaceDimension>;
  using ReducedDimensionTranslationTransformPointer = typename ReducedDimensionTranslationTransformType::Pointer;

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::ParametersType;
  using typename Superclass1::NumberOfParametersType;

  /** Typedef's from TransformBase. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::ParameterMapType;
  using typename Superclass2::RegistrationType;
  using typename Superclass2::CoordRepType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using CombinationTransformType = typename Superclass2::CombinationTransformType;

  /** Typedef SizeType. */
  using SizeType = typename FixedImageType::SizeType;

  /** Execute stuff before the actual registration:
   * \li Set the stack transform parameters.
   * \li Set initial sub transforms.
   * \li Create initial registration parameters.
   */
  int
  BeforeAll() override;

  void
  BeforeRegistration() override;

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  TranslationStackTransform() { this->Superclass1::SetCurrentTransform(m_StackTransform); }

  /** The destructor. */
  ~TranslationStackTransform() override = default;

private:
  elxOverrideGetSelfMacro;

  void
  InitializeTransform();

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** The deleted copy constructor and assignment operator. */
  /** Typedef for stack transform. */
  using StackTransformType = itk::TranslationStackTransform<SpaceDimension>;

  /** The Translation stack transform. */
  const typename StackTransformType::Pointer m_StackTransform{ StackTransformType::New() };

  /** Dummy sub transform to be used to set sub transforms of stack transform. */
  ReducedDimensionTranslationTransformPointer m_DummySubTransform;

  /** Stack variables. */
  unsigned int m_NumberOfSubTransforms;
  double       m_StackOrigin, m_StackSpacing;

  unsigned int
  InitializeTranslationTransform();
};


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTranslationStackTransform.hxx"
#endif

#endif // end #ifndef elxTranslationStackTransform_h
