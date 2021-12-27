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

#ifndef elxReducedDimensionBSplineInterpolator_hxx
#define elxReducedDimensionBSplineInterpolator_hxx

#include "elxReducedDimensionBSplineInterpolator.h"

namespace elastix
{

/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
ReducedDimensionBSplineInterpolator<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Read the desired spline order from the parameter file. */
  unsigned int splineOrder = 1;
  this->GetConfiguration()->ReadParameter(
    splineOrder, "BSplineInterpolationOrder", this->GetComponentLabel(), level, 0);

  /** Check. */
  if (splineOrder == 0)
  {
    xl::xout["warning"] << "WARNING: the BSplineInterpolationOrder is set to 0.\n"
                        << "         It is not possible to take derivatives with this setting.\n"
                        << "         Make sure you use a derivative free optimizer." << std::endl;
  }

  /** Set the splineOrder. */
  this->SetSplineOrder(splineOrder);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxReducedDimensionBSplineInterpolator_hxx
