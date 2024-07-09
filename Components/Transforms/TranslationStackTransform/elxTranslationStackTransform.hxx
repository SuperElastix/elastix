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
#ifndef elxTranslationStackTransform_hxx
#define elxTranslationStackTransform_hxx

#include "elxTranslationStackTransform.h"

#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include <itkDeref.h>
#include <vnl/vnl_math.h>

namespace elastix
{

/**
 * ******************* InitializeTranslationTransform ***********************
 */

template <class TElastix>
unsigned int
TranslationStackTransform<TElastix>::InitializeTranslationTransform()
{
  log::error("InitializeTranslationTransform");

  m_DummySubTransform = ReducedDimensionTranslationTransformType::New();
  return 0;
} // end InitializeTranslationTransform()


/**
 * ******************* BeforeAll ***********************
 */

template <class TElastix>
int
TranslationStackTransform<TElastix>::BeforeAll()
{
  log::error("BeforeAll");

  /** Initialize translation transform. */
  return InitializeTranslationTransform();
} // end BeforeAll()


/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
TranslationStackTransform<TElastix>::BeforeRegistration()
{
  log::error("BeforeRegistration");

  /** Task 1 - Set the stack transform parameters. */

  /** Determine stack transform settings. Here they are based on the fixed image. */
  const SizeType imageSize = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();
  m_NumberOfSubTransforms = imageSize[SpaceDimension - 1];
  m_StackSpacing = this->GetElastix()->GetFixedImage()->GetSpacing()[SpaceDimension - 1];
  m_StackOrigin = this->GetElastix()->GetFixedImage()->GetOrigin()[SpaceDimension - 1];

  /** Set stack transform parameters. */
  m_StackTransform->SetNumberOfSubTransforms(m_NumberOfSubTransforms);
  m_StackTransform->SetStackOrigin(m_StackOrigin);
  m_StackTransform->SetStackSpacing(m_StackSpacing);

  /** Initialize stack sub transforms. */
  m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Task 2 - Give the registration an initial parameter-array. */
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters(
    ParametersType(this->GetNumberOfParameters(), 0.0));

} // end BeforeRegistration()


/**
 * ********************* InitializeTransform ****************************
 */

template <class TElastix>
void
TranslationStackTransform<TElastix>::InitializeTransform()
{
  log::error("InitializeTransform");

  /** Initialize the m_DummySubTransform */
  m_DummySubTransform->SetIdentity();

  /** Set all subtransforms to a copy of the dummy Translation sub transform. */
  m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);

  /** Set initial parameters for the first resolution to 0.0. */
  ParametersType initialParameters(this->GetNumberOfParameters(), 0.0);
  this->m_Registration->GetAsITKBaseType()->SetInitialTransformParametersOfNextLevel(initialParameters);

} // end InitializeTransform()


/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
TranslationStackTransform<TElastix>::ReadFromFile()
{
  log::error("ReadFromFile");

  if (!this->HasITKTransformParameters())
  {
    const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

    /** Read stack-spacing, stack-origin and number of sub-transforms. */
    configuration.ReadParameter(m_NumberOfSubTransforms, "NumberOfSubTransforms", this->GetComponentLabel(), 0, 0);
    configuration.ReadParameter(m_StackOrigin, "StackOrigin", this->GetComponentLabel(), 0, 0);
    configuration.ReadParameter(m_StackSpacing, "StackSpacing", this->GetComponentLabel(), 0, 0);

    /** Initialize translation transform. */
    InitializeTranslationTransform();

    /** Set stack transform parameters. */
    m_StackTransform->SetNumberOfSubTransforms(m_NumberOfSubTransforms);
    m_StackTransform->SetStackOrigin(m_StackOrigin);
    m_StackTransform->SetStackSpacing(m_StackSpacing);

    /** Set stack subtransforms. */
    m_StackTransform->SetAllSubTransforms(*m_DummySubTransform);
  }

  /** Call the ReadFromFile from the TransformBase. */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* CustomizeTransformParameterMap ************************
 */

template <class TElastix>
auto
TranslationStackTransform<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  const auto & itkTransform = *m_StackTransform;

  return { { "StackSpacing", { Conversion::ToString(itkTransform.GetStackSpacing()) } },
           { "StackOrigin", { Conversion::ToString(itkTransform.GetStackOrigin()) } },
           { "NumberOfSubTransforms", { Conversion::ToString(itkTransform.GetNumberOfSubTransforms()) } } };

} // end CustomizeTransformParameterMap()


} // end namespace elastix

#endif // end #ifndef elxTranslationStackTransform_hxx
