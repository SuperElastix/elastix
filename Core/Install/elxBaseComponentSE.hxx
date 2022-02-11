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

#ifndef elxBaseComponentSE_hxx
#define elxBaseComponentSE_hxx

#include "elxBaseComponentSE.h"
#include "itkObject.h"

namespace elastix
{

/**
 * *********************** SetElastix ***************************
 */

template <class TElastix>
void
BaseComponentSE<TElastix>::SetElastix(TElastix * const _arg)
{
  /** If this->m_Elastix is not set, then set it. */
  if (this->m_Elastix != _arg)
  {
    this->m_Elastix = _arg;

    if (_arg != nullptr)
    {
      this->m_Configuration = _arg->GetConfiguration();
      this->m_Registration = _arg->GetElxRegistrationBase();
    }
    this->GetSelf().Modified();
  }

} // end SetElastix


/**
 * *********************** SetConfiguration ***************************
 *
 * Added for transformix.
 */

template <class TElastix>
void
BaseComponentSE<TElastix>::SetConfiguration(Configuration * const _arg)
{
  /** If this->m_Configuration is not set, then set it.*/
  if (this->m_Configuration != _arg)
  {
    this->m_Configuration = _arg;
    this->GetSelf().Modified();
  }

} // end SetConfiguration


} // end namespace elastix

#endif // end #ifndef elxBaseComponentSE_hxx
