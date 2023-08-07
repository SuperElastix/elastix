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

#ifndef elxExternalTransform_hxx
#define elxExternalTransform_hxx

#include "elxExternalTransform.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template <class TElastix>
ExternalTransform<TElastix>::ExternalTransform()
{
  Superclass1::SetCurrentTransform(m_AdvancedTransformAdapter);
}

/**
 * ************************* ReadFromFile ************************
 */

template <class TElastix>
void
ExternalTransform<TElastix>::ReadFromFile()
{
  /** Call the ReadFromFile from the TransformBase. */
  Superclass2::ReadFromFile();

  const Configuration & configuration = Deref(Superclass2::GetConfiguration());

  if (const auto objectPtr =
        configuration.RetrieveParameterValue<const itk::Object *>(nullptr, "TransformAddress", 0, false))
  {
    if (const auto transform = dynamic_cast<const typename AdvancedTransformAdapterType::TransformType *>(objectPtr))
    {
      m_AdvancedTransformAdapter->SetExternalTransform(transform);
    }
    else
    {
      itkExceptionMacro("The specified TransformAddress is not the address of the correct transform type!");
    }
  }
  else
  {
    m_AdvancedTransformAdapter->SetExternalTransform(nullptr);
  }
}


/**
 * ************************* CreateDerivedTransformParameterMap ************************
 */

template <class TElastix>
auto
ExternalTransform<TElastix>::CreateDerivedTransformParameterMap() const -> ParameterMapType
{
  return { { "TransformAddress",
             { Conversion::ObjectPtrToString(m_AdvancedTransformAdapter->GetExternalTransform()) } } };
}

} // namespace elastix

#endif
