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
#ifndef elxBSplineResampleInterpolator_hxx
#define elxBSplineResampleInterpolator_hxx

#include "elxBSplineResampleInterpolator.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
BSplineResampleInterpolator<TElastix>::BeforeRegistration()
{
  /** BSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  this->m_Configuration->ReadParameter(splineOrder, "FinalBSplineInterpolationOrder", 0);

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder(splineOrder);

} // end BeforeRegistration()


/**
 * ******************* ReadFromFile  ****************************
 */

template <class TElastix>
void
BSplineResampleInterpolator<TElastix>::ReadFromFile()
{
  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();

  /** BSplineResampleInterpolator specific. */

  /** Set the SplineOrder, default = 3. */
  unsigned int splineOrder = 3;

  /** Read the desired splineOrder from the parameterFile. */
  this->m_Configuration->ReadParameter(splineOrder, "FinalBSplineInterpolationOrder", 0);

  /** Set the splineOrder in the superclass. */
  this->SetSplineOrder(splineOrder);

} // end ReadFromFile()


/**
 * ******************* CreateDerivedTransformParametersMap ******************************
 */

template <class TElastix>
auto
BSplineResampleInterpolator<TElastix>::CreateDerivedTransformParametersMap() const -> ParameterMapType
{
  return { { "FinalBSplineInterpolationOrder", { Conversion::ToString(this->GetSplineOrder()) } } };

} // end CreateDerivedTransformParametersMap()


} // end namespace elastix

#endif // end #ifndef elxBSplineResampleInterpolator_hxx
